import json
import os
import shutil
import statistics
import traceback
from dataclasses import InitVar, dataclass, field
from random import Random
from typing import Literal, Type, cast

import numpy as np
import pandas as pd
from fantasy_py import (
    CacheMode,
    ContestStyle,
    DataNotAvailableException,
    FantasyException,
    GameScheduleEpoch,
    SeasonPart,
    SlateDict,
    cache_to_file,
    db,
    dt_to_filename_str,
    log,
)
from fantasy_py.inference import ImputeFailure, get_models_by_name
from fantasy_py.lineup import (
    DEEP_LINEUP_POSITION_REMAPPINGS,
    FantasyCostAggregate,
    FantasyService,
    LineupGenerationFailure,
    gen_lineups,
)
from fantasy_py.lineup.knapsack import MixedIntegerKnapsackSolver
from fantasy_py.sport import DateNotAvailableError, Starters
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


def __get_max_slate_games_cache_filename(
    session: Session, season: int, style: ContestStyle, service_cls: FantasyService
):
    filename_parts = [
        "max_slate_games",
        session.info["fantasy.db_manager"].ABBR,
        str(season),
        service_cls.SERVICE_NAME,
        style.name,
    ]
    return "-".join(filename_parts) + ".cache"


@cache_to_file(filename_func=__get_max_slate_games_cache_filename)
def _get_max_slate_games(
    session: Session, season: int, style: ContestStyle, service_cls: FantasyService
):
    """return tuple of the min and max number of games in slates in the requested season"""
    max_games = 0

    for slate in session.query(db.DailyFantasySlate).filter(
        db.DailyFantasySlate.season == season,
        db.DailyFantasySlate._style == style.name,
        db.DailyFantasySlate.service == service_cls.SERVICE_NAME,
    ):
        games_count = len(slate.games)
        max_games = max(max_games, games_count)

    return max_games


def export(
    db_obj: db.FantasySQLAlchemyWrapper,
    dataset_name,
    parent_dest_dir: str,
    requested_seasons: tuple[int, int] | int | None,
    slate_games_range: tuple[int, int] | None,
    case_count: int,
    existing_files_mode: ExistingFilesMode,
    service_cls: FantasyService,
    style: ContestStyle,
    seed,
    cache_dir,
    cache_mode: CacheMode,
):
    """
    requested_seasons: tuple of (start, end) seasons, inclusive, to choose from,\
        or a single seasons, or None for all seasons
    """
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

    if requested_seasons is None or not isinstance(requested_seasons, int):
        seasons = list(db_obj.db_manager.get_seasons())
        if requested_seasons is not None:
            seasons = [
                season
                for season in seasons
                if requested_seasons[0] <= season <= requested_seasons[1]
            ]
    else:
        seasons = [requested_seasons]
        if requested_seasons not in db_obj.db_manager.get_seasons():
            raise ExportError(
                f"Requested season {requested_seasons} is not supported for this sport"
            )
    if len(seasons) == 0:
        raise ExportError(f"Requested seasons {requested_seasons} resulted in no seasons selected")

    with db_obj.session_scoped() as session:
        if slate_games_range is None:
            _LOGGER.info("Getting max games for slates using season=%i", max(seasons))
            slate_games_range = 2, _get_max_slate_games(
                session,
                max(seasons),
                style,
                service_cls,
                cache_dir=cache_dir,
                cache_mode=cache_mode,
            )

        _LOGGER.info(
            "Starting exporting of n=%i cases for sport=%s seasons=%s slate_games_range=%s to '%s'",
            case_count,
            db_obj.db_manager.ABBR,
            seasons,
            slate_games_range,
            dataset_dest_dir,
        )

        rand_obj = _RandomSlateSelector(
            session,
            seasons,
            slate_games_range,
            service_cls.SERVICE_NAME,
            style,
            seed,
            cache_dir,
            cache_mode,
        )
        expected_cols: list[str] | None = None
        for sample_num in (prog_iter := tqdm(range(case_count), desc=dataset_name)):
            for attempt_num in range(_MAX_SLATE_ATTEMPTS):
                total_attempts += 1
                slate_def = rand_obj.next
                _LOGGER.info(
                    "Sample #%i try #%i: %s '%s' (%i games)",
                    sample_num + 1,
                    attempt_num + 1,
                    slate_def.epoch,
                    slate_def.slate,
                    len(slate_def.game_descs),
                )
                prog_iter.set_postfix(attempts=total_attempts)
                try:
                    df = _get_slate_sample(
                        db_obj,
                        service_cls,
                        slate_def,
                        model_names,
                        style,
                        seed,
                        cache_dir,
                        cache_mode,
                    )
                    if expected_cols is None:
                        expected_cols = sorted(df.columns)
                        _LOGGER.info("Expected sample cols set to: %s", expected_cols)
                    elif (df_cols := sorted(df.columns)) != expected_cols:
                        raise ExportError(
                            f"Unexpected cols found For sample #{sample_num + 1}, "
                            f"expected={expected_cols} found={df_cols}"
                        )
                    break
                except _GenLineupHelperFailure as ex:
                    _LOGGER.warning(
                        "Attempt %i for sample %i failed to create a sample "
                        "for %i-%i game_ids=%s ERROR: %s",
                        attempt_num + 1,
                        sample_num + 1,
                        slate_def.epoch.season,
                        slate_def.epoch.game_number,
                        slate_def.game_descs,
                        ex.original_ex,
                    )
                    failed_attempts.append(slate_def)
            else:
                raise ExportError(
                    f"Failed to create a slate for sample #{sample_num + 1} "
                    f"after {attempt_num + 1} attempts"
                )
            successful_attempts.append(slate_def)
            filename = f"{slate_def.epoch.season}-{slate_def.epoch.game_number}-{slate_def.hash_value}.parquet"
            dataset_filepath = os.path.join(dataset_dest_dir, filename)
            df.to_parquet(dataset_filepath)
            _LOGGER.success(
                "Sample #%i successfully created. Written to '%s'", sample_num + 1, dataset_filepath
            )
            df_lens.append(len(df))
            samples_meta.append({**slate_def.meta_data, "player_count": len(df)})

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
                "style": style.value,
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


@dataclass
class SlateDef:
    """a slate definition, used to infer a sample"""

    epoch: GameScheduleEpoch
    starters: Starters
    slate: str
    game_descs: list[str]
    hash_value: str

    @property
    def slate_info(self):
        assert self.starters.slates is not None
        return self.starters.slates[self.slate]

    @property
    def meta_data(self):
        return {
            "season": self.epoch.season,
            "game_number": self.epoch.game_number,
            "game_descs": self.game_descs,
            "slate": self.slate,
            "date": self.epoch.date.isoformat(),
            "games_hash": self.hash_value,
        }


def _pick_slate(starters: Starters):
    """pick the slate with the most games"""
    max_slate: tuple[None | str, int] = (None, 0)
    assert starters.slates is not None
    for slate, slate_info in starters.slates.items():
        if (games_count := len(slate_info.get("games", []))) > max_slate[1]:
            max_slate = slate, games_count
    assert max_slate[0] is not None
    return max_slate


def __gen_lineup_helper_cache_filename(
    _: db.FantasySQLAlchemyWrapper,
    fca: FantasyCostAggregate,
    model_names: list[str],
    service_cls: Type[FantasyService],
    epoch: GameScheduleEpoch,
    seed,
    style: ContestStyle,
    cost_id,
):
    assert fca.games is not None
    game_ids = sorted(game['id'] for game in fca.games)
    hash_val = (sorted(model_names), seed, style, cost_id, game_ids)
    hashed_val = constant_hasher(hash_val)
    filename = (
        f"dd-gen_lineup-{fca.sport_abbr}-{epoch.date.isoformat().replace('-', '')}-"
        f"{service_cls.ABBR}-{hashed_val}.cache"
    )
    return filename


class _GenLineupHelperFailure(FantasyException):
    """returned if the helper fails, picklable to be compatible with file cacher"""

    def __init__(self, original_ex: FantasyException, tb_str: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_ex = original_ex
        self.tb_str = tb_str

    def __reduce__(self):
        """required code pickling during file caching"""
        return _GenLineupHelperFailure, (self.original_ex, self.tb_str)


@cache_to_file(filename_func=__gen_lineup_helper_cache_filename, timeout=60 * 60 * 24 * 7)
def _gen_lineup_helper(
    db_obj: db.FantasySQLAlchemyWrapper,
    fca: FantasyCostAggregate,
    model_names: list[str],
    service_cls: Type[FantasyService],
    epoch: GameScheduleEpoch,
    seed,
    style: ContestStyle,
    cost_id,
    cache_dir=None,
    cache_mode=None,
):
    sport_constraints = service_cls.get_constraints(db_obj.db_manager.ABBR, style=style)
    assert sport_constraints is not None
    solver = MixedIntegerKnapsackSolver(
        sport_constraints.lineup_constraints,
        sport_constraints.budget,
        fill_all_positions=sport_constraints.fill_all_positions,
        random_seed=seed,
    )

    assert cost_id is not None

    new_slate_info = cast(SlateDict, {"style": style.name, "cost_id": cost_id})

    try:
        top_lineup, scores = gen_lineups(
            db_obj,
            fca,
            model_names,
            solver,
            service_cls,
            1,
            slate_info=new_slate_info,
            slate_epoch=epoch,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
            score_data_type="historic",
            scores_to_include=["predicted"],
        )
        return top_lineup, scores
    except (ImputeFailure, DataNotAvailableException, LineupGenerationFailure) as ex:
        return _GenLineupHelperFailure(ex, traceback.format_exc(limit=None, chain=True))


def _get_slate_sample(
    db_obj: db.FantasySQLAlchemyWrapper,
    service_cls: Type[FantasyService],
    slate_def: SlateDef,
    model_names,
    style: ContestStyle,
    seed,
    cache_dir,
    cache_mode,
):
    fca = db_obj.db_manager.fca_from_starters(
        db_obj,
        slate_def.starters,
        service_cls.SERVICE_NAME,
        slate=slate_def.slate,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )

    cost_id = slate_def.slate_info.get("cost_id")
    assert cost_id is not None

    # use historic data to generate the best possible lineup
    gen_lineup_result = _gen_lineup_helper(
        db_obj,
        fca,
        model_names,
        service_cls,
        slate_def.epoch,
        seed,
        style,
        cost_id,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )

    if isinstance(gen_lineup_result, _GenLineupHelperFailure):
        raise gen_lineup_result

    top_lineup, scores = gen_lineup_result

    hist_df = scores["historic"].rename(columns={"fpts": "fpts-historic"})
    pred_df = scores["predicted"].rename(columns={"fpts": "fpts-predicted"})

    if hist_df.player_id.isna().any():
        hist_df.fillna(value={"player_id": -1}, inplace=True)
    if pred_df.player_id.isna().any():
        pred_df.fillna(value={"player_id": -1}, inplace=True)
    score_df = hist_df.join(
        pred_df.set_index(["game_id", "team_id", "player_id"]),
        on=["game_id", "team_id", "player_id"],
    )
    score_df.replace({"player_id": -1}, np.nan, inplace=True)

    pos_mapping = DEEP_LINEUP_POSITION_REMAPPINGS.get(db_obj.db_manager.ABBR, {})

    def cost_func(row):
        if "player_id" not in row or pd.isna(row.player_id):
            pt_dict = fca.get_mi_team(row.team_id)
        else:
            pt_dict = fca.get_mi_player(row.player_id)

        assert isinstance(pt_dict, dict) and "positions" in pt_dict
        dict_ = {"pos:" + pos_mapping.get(pos, pos): 1 for pos in pt_dict["positions"]}

        assert isinstance(pt_dict, dict) and "cost" in pt_dict and isinstance(pt_dict["cost"], dict)
        pt_cost_dict = pt_dict["cost"][service_cls.SERVICE_NAME]
        assert isinstance(pt_cost_dict, dict)
        dict_["cost"] = pt_cost_dict.get(cost_id)
        return dict_

    cost_pos_df = score_df.apply(cost_func, axis=1, result_type="expand")

    def addl_data_func(row):
        ret = {}
        if pd.isna(row.player_id):
            ret["in-lineup"] = row.team_id in top_lineup[0].team_ids
        else:
            ret["in-lineup"] = row.player_id in top_lineup[0].player_ids
        if pd.isna(row.player_id):
            ret["opp_id"] = fca.get_mi_team(row.team_id)["opp_id"]
        else:
            ret["opp_id"] = fca.get_mi_player(row.player_id)["opp_id"]

        return ret

    addl_df = score_df.apply(addl_data_func, axis=1, result_type="expand")

    df = pd.concat([score_df.drop("game_id", axis=1), cost_pos_df, addl_df], axis=1).query(
        "cost.notna()"
    )

    pos_cols_to_drop: list[str] = []
    for col in df.columns:
        if not col.startswith("pos:"):
            continue
        if df[col].isna().all():
            pos_cols_to_drop.append(col)
            continue
        df[col] = df[col].fillna(0)

    if len(pos_cols_to_drop) > 0:
        df.drop(columns=pos_cols_to_drop, inplace=True)
    return df


@dataclass
class _RandomSlateSelector:
    session: Session
    seasons: list[int]
    """list of seasons to select slates from"""
    slate_games_range: tuple[int, int]
    service_name: str
    style: ContestStyle
    seed: InitVar[int]
    cache_dir: str
    cache_mode: CacheMode
    season_parts: list[SeasonPart] = field(default_factory=lambda: [SeasonPart.REGULAR])

    _rand_obj: Random = field(init=False)
    _past_selections: set[int] = field(init=False, default_factory=set)
    """ past randomly generated slate selections in the form of a 
    set of sorted tuples of game IDs"""

    def __post_init__(self, seed):
        self._rand_obj = Random(seed)

    @property
    def next(self):
        """
        return what the next slate will be based on as a tuple of (season, game_num, game_ids)
        """
        for attempt in range(_MAX_NEXT_FAILURES):
            season = self._rand_obj.choice(self.seasons)
            game_num = self._rand_obj.randint(
                1, self.session.info["fantasy.db_manager"].get_max_epochs(season)
            )
            game_count = self._rand_obj.randint(*self.slate_games_range)

            try:
                epoch = cast(
                    GameScheduleEpoch,
                    self.session.info["fantasy.db_manager"].epoch_for_game_number(season, game_num),
                )
            except DateNotAvailableError:
                _LOGGER.info(
                    "Skipping sample candidate %i season=%i, game_num=%i: date not available",
                    attempt + 1,
                    season,
                    game_num,
                )
                continue
            if epoch.season_part not in self.season_parts:
                _LOGGER.info(
                    "Skipping sample candidate %i epoch=%s: season part %s not in %s",
                    attempt + 1,
                    epoch,
                    epoch.season_part.name,
                    [part.name for part in self.season_parts],
                )
                continue
            no_data_ex = None
            try:
                starters = cast(
                    Starters,
                    self.session.info["fantasy.db_manager"].get_starters(
                        self.service_name,
                        games_date=epoch.date,
                        db_obj=self.session.info["db_obj"],
                        remote_allowed=False,
                        style=self.style,
                        cache_dir=self.cache_dir,
                        cache_mode=self.cache_mode,
                    ),
                )
            except DataNotAvailableException as ex:
                starters = None
                no_data_ex = ex

            if starters is None:
                _LOGGER.info(
                    "Skipping sample candidate %i epoch=%s: ex=%s", attempt + 1, epoch, no_data_ex
                )
                continue

            assert starters.slates is not None
            slate_with_most_games, slate_games_count = _pick_slate(starters)
            if game_count > slate_games_count:
                _LOGGER.info(
                    "Skipping sample candidate %i epoch=%s: game_count=%i is too high "
                    "for largest slate with %i games",
                    attempt + 1,
                    epoch,
                    game_count,
                    slate_games_count,
                )
                continue

            available_slate_games = starters.slates[slate_with_most_games].get("games")
            assert available_slate_games is not None
            game_descs = sorted(self._rand_obj.sample(available_slate_games, game_count))
            filtered_starters = starters.filter_by_slate(
                slate_with_most_games, game_descs=game_descs
            )

            if (games_hash := cast(int, constant_hasher(game_descs))) in self._past_selections:
                _LOGGER.info(
                    "Skipping sample candidate %i epoch=%s: "
                    "random slate games sample already used. %s",
                    attempt + 1,
                    epoch,
                    game_descs,
                )
                continue

            self._past_selections.add(games_hash)
            break
        else:
            raise ExportError(
                f"Failed to find a viable slate candidate for {season} "
                "after {_MAX_NEXT_FAILURES} failures"
            )

        return SlateDef(
            epoch, filtered_starters, slate_with_most_games, game_descs, str(games_hash)
        )
