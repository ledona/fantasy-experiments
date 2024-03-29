import json
import os
import shutil
import statistics
from random import Random
from typing import Literal, NamedTuple, Type, cast

import pandas as pd
from fantasy_py import (
    CacheMode,
    ContestStyle,
    DataNotAvailableException,
    FantasyException,
    db,
    dt_to_filename_str,
    log,
)
from fantasy_py.inference import ImputeFailure, get_models_by_name
from fantasy_py.lineup import FantasyService, gen_lineups
from fantasy_py.lineup.knapsack import MixedIntegerKnapsackSolver
from fantasy_py.sport import Starters
from ledona import constant_hasher
from sqlalchemy.orm import Session
from tqdm import tqdm

_LOGGER = log.get_logger(__name__)
_MAX_NEXT_FAILURES = 100
"""maximum number of failed next search tries before giving up"""
_MAX_SLATE_ATTEMPTS = 20
"""max number of failed attempts to create a valid slate for an epoch"""

ExistingFilesMode = Literal["append", "delete", "fail"]


class ExportError(FantasyException):
    """raised when there is an error exporting data"""


def _prep_dataset_directory(
    parent_dir: str, dataset_name: str | None, existing_files_mode: ExistingFilesMode
):
    if not os.path.exists(parent_dir):
        _LOGGER.info("Creating parent destination directory '%s'", parent_dir)
        os.mkdir(parent_dir)
    elif os.path.isfile(parent_dir):
        raise ExportError(f"Parent destination directory '{parent_dir}' is a file")

    dataset_dest_dir = os.path.join(parent_dir, dataset_name or dt_to_filename_str())
    if not os.path.exists(dataset_dest_dir):
        _LOGGER.info("Creating dataset destination directory '%s'", dataset_dest_dir)
        os.mkdir(dataset_dest_dir)
    elif os.path.isfile(dataset_dest_dir):
        raise ExportError(f"Dataset destination directory '{dataset_dest_dir}' is a file")
    elif existing_files_mode == "fail":
        raise ExportError(f"Dataset directory '{dataset_dest_dir}' already exists")
    elif existing_files_mode == "delete":
        _LOGGER.info("Deleting/recreating existing dataset directory '%s'", dataset_dest_dir)
        shutil.rmtree(dataset_dest_dir)
        os.mkdir(dataset_dest_dir)
    elif existing_files_mode == "append":
        _LOGGER.info(
            "Appending export data to anything already in existing dataset export path '%s'",
            dataset_dest_dir,
        )
    else:
        raise ValueError(f"Unknown existing_files_mode '{existing_files_mode}'")

    return dataset_dest_dir


_POS_REMAP = {"nhl": {"RW": "W", "LW": "W", "R": "W", "L": "W"}}
"""mapping for sport -> position-mapping-dict"""


def export(
    db_obj: db.FantasySQLAlchemyWrapper,
    dataset_name,
    parent_dest_dir: str,
    requested_seasons: list[int] | None,
    slate_games_range: tuple[int, int],
    case_count: int,
    existing_files_mode: ExistingFilesMode,
    service_cls: FantasyService,
    style: ContestStyle,
    seed,
    cache_dir,
    cache_mode: CacheMode,
):
    _LOGGER.info("Starting export")
    model_names = service_cls.DEFAULT_MODEL_NAMES.get(db_obj.db_manager.ABBR)
    if model_names is None or len(model_names) == 0:
        raise ExportError(f"Sport '{db_obj.db_manager.ABBR}' has no default models")

    dataset_dest_dir = _prep_dataset_directory(parent_dest_dir, dataset_name, existing_files_mode)

    samples_meta: list[dict] = []
    total_attempts = 0
    successful_attempts = []
    failed_attempts = []
    df_lens = []
    with db_obj.session_scoped() as session:
        if requested_seasons is None:
            seasons = [cast(int, row[0]) for row in session.query(db.Game.season).distinct().all()]
        else:
            if not set(requested_seasons).issubset(db_obj.db_manager.get_seasons()):
                raise ExportError(f"Requested seasons {requested_seasons} not supported by sport")
            seasons = requested_seasons

        _LOGGER.info(
            "Exporting n=%i for sport=%s seasons=%s to '%s'",
            case_count,
            db_obj.db_manager.ABBR,
            seasons,
            dataset_dest_dir,
        )

        rand_obj = _RandomSlateSelector(session, seasons, slate_games_range, seed)
        for sample_num in (prog_iter := tqdm(range(case_count), desc="sample-slates")):
            for attempt_num in range(_MAX_SLATE_ATTEMPTS):
                total_attempts += 1
                slate_def = rand_obj.next
                prog_iter.set_postfix(attempts=total_attempts)
                sample_meta = slate_def._asdict()

                try:
                    df = _get_slate_sample(
                        db_obj,
                        service_cls,
                        slate_def.game_ids,
                        model_names,
                        cache_dir,
                        cache_mode,
                        db_obj.db_manager.epoch_for_game_number(
                            slate_def.season, slate_def.game_number
                        ),
                        seed,
                        style,
                    )
                    break
                except (ImputeFailure, DataNotAvailableException) as ex:
                    _LOGGER.warning(
                        "Attempt %i for sample %i failed to create a slate "
                        "for %i-%i game_ids=%s: %s",
                        attempt_num + 1,
                        sample_num + 1,
                        slate_def.season,
                        slate_def.game_number,
                        slate_def.game_ids,
                        ex,
                    )
                    failed_attempts.append(slate_def)
            else:
                raise ExportError(
                    f"Failed to create a slate for sample #{sample_num + 1} "
                    f"after {attempt_num + 1} attempts"
                )
            successful_attempts.append(slate_def)
            filename = f"{slate_def.season}-{slate_def.game_number}-{slate_def.game_ids_hash}.pq"
            df.to_parquet(os.path.join(dataset_dest_dir, filename))
            sample_meta["items"] = len(df)
            df_lens.append(len(df))
            samples_meta.append(sample_meta)

    models_dict = {
        model.name: model.dt_trained.isoformat()
        for model in get_models_by_name(model_names)
        if model.dt_trained is not None
    }
    with open(meta_filepath := os.path.join(dataset_dest_dir, "samples_meta.json"), "w") as f_:
        json.dump(
            {
                "models": models_dict,
                "sport": db_obj.db_manager.ABBR,
                "seasons": seasons,
                "seed": seed,
                "samples": samples_meta,
                "service": service_cls.SERVICE_NAME,
                "style": style.name,
            },
            f_,
            indent="\t",
        )
    _LOGGER.info(
        "Meta data written to '%s' min-median-max df lengths = %i %i %i",
        meta_filepath,
        min(df_lens),
        statistics.median(df_lens),
        max(df_lens),
    )


def _get_slate_sample(
    db_obj: db.FantasySQLAlchemyWrapper,
    service_cls: Type[FantasyService],
    game_ids: list[int],
    model_names,
    cache_dir,
    cache_mode,
    epoch,
    seed,
    style: ContestStyle,
):
    starters = Starters.from_historic_data(
        db_obj, service_cls.SERVICE_NAME, epoch, game_ids=game_ids, style=style
    )
    assert starters is not None and starters.slates is not None and len(starters.slates) == 1
    slate_info = next(iter(starters.slates.values()))
    fca = db_obj.db_manager.fca_from_starters(db_obj, starters, service_cls.SERVICE_NAME)
    sport_constraints = service_cls.get_constraints(db_obj.db_manager.ABBR, style=style)
    assert sport_constraints is not None
    solver = MixedIntegerKnapsackSolver(
        sport_constraints.lineup_constraints,
        sport_constraints.budget,
        fill_all_positions=sport_constraints.fill_all_positions,
        random_seed=seed,
    )

    # use historic data to generate the best possible lineup
    top_lineup, scores = gen_lineups(
        db_obj,
        fca,
        model_names,
        solver,
        service_cls,
        1,
        slate_info=slate_info,
        slate_epoch=epoch,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
        score_data_type="historic",
        scores_to_include=["predicted"],
    )

    hist_df = scores["historic"].rename(columns={"fpts": "fpts-historic"})
    pred_df = scores["predicted"].rename(columns={"fpts": "fpts-predicted"})
    score_df = hist_df.join(
        pred_df.set_index(["game_id", "team_id", "player_id"]),
        on=["game_id", "team_id", "player_id"],
    )
    pos_mapping = _POS_REMAP.get(db_obj.db_manager.ABBR, {})

    def cost_func(row):
        if "player_id" not in row or pd.isna(row.player_id):
            pt_dict = fca.get_mi_team(row.team_id)
        else:
            pt_dict = fca.get_mi_player(row.player_id)
        dict_ = {
            "cost": pt_dict["cost"][service_cls.SERVICE_NAME][slate_info["cost_id"]],
        }
        for pos in pt_dict["positions"]:
            dict_["pos:" + pos_mapping.get(pos, pos)] = 1
        return dict_

    cost_pos_df = score_df.apply(cost_func, axis=1, result_type="expand")

    def addl_data_func(row):
        ret = {}
        if pd.isna(row.player_id) and row.team_id in top_lineup[0].team_ids:
            ret["in-lineup"] = 1
        else:
            ret["in-lineup"] = row.player_id in top_lineup[0].player_ids
        if pd.isna(row.player_id):
            ret["opp_id"] = fca.get_mi_team(row.team_id)["opp_id"]
        else:
            ret["opp_id"] = fca.get_mi_player(row.player_id)["opp_id"]

        return ret

    addl_df = score_df.apply(addl_data_func, axis=1, result_type="expand")

    df = pd.concat([score_df.drop("game_id", axis=1), cost_pos_df, addl_df], axis=1)

    for col in df.columns:
        if not col.startswith("pos:"):
            continue
        df[col] = df[col].fillna(0)
    return df


class _RandomSlateSelector:
    def __init__(
        self, session: Session, seasons: list[int], slate_games_range: tuple[int, int], seed
    ):
        self._session = session
        self._seasons = seasons
        self._slate_games_range = slate_games_range
        self._rand_obj = Random(seed)

        self._past_selections: set[int] = set()
        """ past randomly generated slate selections in the form of a 
        set of sorted tuples of game IDs"""

    class NextSlateDef(NamedTuple):
        season: int
        game_number: int
        game_ids: list[int]
        game_ids_hash: int

    @property
    def next(self):
        """
        return what the next slate will be based on as a tuple of (season, game_num, game_ids)
        """
        season = self._rand_obj.choice(self._seasons)
        for _ in range(_MAX_NEXT_FAILURES):
            game_num = self._rand_obj.randint(
                1, self._session.info["fantasy.db_manager"].get_max_epochs(season)
            )
            all_game_ids = list(
                cast(int, row[0])
                for row in self._session.query(db.Game.id)
                .filter(db.Game.season == season, db.Game.game_number == game_num)
                .all()
            )
            game_count = self._rand_obj.randint(*self._slate_games_range)
            if game_count > len(all_game_ids):
                continue
            game_ids = sorted(self._rand_obj.sample(all_game_ids, game_count))
            if (game_ids_hash := cast(int, constant_hasher(game_ids))) in self._past_selections:
                continue

            self._past_selections.add(game_ids_hash)
            break
        else:
            raise ExportError(
                f"After {_MAX_NEXT_FAILURES} tries, failed to find a slate for {season}"
            )

        return self.NextSlateDef(season, game_num, game_ids, game_ids_hash)
