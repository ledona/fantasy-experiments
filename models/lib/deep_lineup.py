"""functions required for exporting and training deep models"""

import argparse
import json
import os
import shlex
import shutil
from random import Random
from typing import Literal, cast

from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CacheMode,
    CLSRegistry,
    FantasyException,
    db,
    dt_to_filename_str,
    log,
)
from fantasy_py.lineup import gen_predictions_for_hypothetical_games
from ledona import constant_hasher
from sqlalchemy.orm import Session
from tqdm import tqdm

_DEFAULT_SAMPLE_COUNT = 10
_DEFAULT_PARENT_DATASET_PATH = "/fantasy-isync/fantasy-modeling/deep_lineup"
_DEFAULT_GAMES_PER_SLATE = 4, 6
_LOGGER = log.get_logger(__name__)


class ExportError(FantasyException):
    """raised when there is an error exporting data"""


_ExistingFilesMode = Literal["append", "delete", "fail"]


def _prep_dataset_directory(
    parent_dir: str, dataset_name: str | None, existing_files_mode: _ExistingFilesMode
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


def _get_slate_df(session: Session, game_ids: list[int], model_names, cache_dir, cache_mode):
    games = [
        game.to_game_dict(include_players=True)
        for game in session.query(db.Game).filter(db.Game.id.in_(game_ids)).all()
    ]

    predictions = gen_predictions_for_hypothetical_games(
        session,
        model_names,
        games,
        cache_dir=cache_dir,
        cache_mode=cache_mode,
    )[0]

    raise NotImplementedError("""
        1. find top score for lineups based on games
        2. transform predictions to train/test data format 
    """)


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

    @property
    def next(self):
        """
        return what the next slate will be based on as a tuple of (season, game_num, game_ids)
        """
        season = self._rand_obj.choice(self._seasons)
        while True:
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
            if (game_ids_hash := constant_hasher(game_ids)) in self._past_selections:
                continue

            self._past_selections.add(game_ids_hash)
            break

        return season, game_num, game_ids, game_ids_hash


def _export_deep_dataset(
    db_obj: db.FantasySQLAlchemyWrapper,
    dataset_name,
    parent_dest_dir: str,
    requested_seasons: list[int] | None,
    slate_games_range: tuple[int, int],
    case_count: int,
    existing_files_mode: _ExistingFilesMode,
    model_names: list[str],
    seed,
    cache_dir,
    cache_mode: CacheMode,
):
    _LOGGER.info("Starting export")
    dataset_dest_dir = _prep_dataset_directory(parent_dest_dir, dataset_name, existing_files_mode)
    if requested_seasons is None:
        seasons = cast(list[int], db_obj.db_manager.get_seasons())
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

    samples_meta: list[dict] = []

    with db_obj.session_scoped() as session:
        rand_obj = _RandomSlateSelector(session, seasons, slate_games_range, seed)
        for _ in (prog_iter := tqdm(range(case_count))):
            season, game_number, game_ids, game_ids_hashed = rand_obj.next

            prog_iter.set_postfix_str(f"{season}-{game_number}")
            sample_meta = {"season": season, "game_num": game_number, "game_ids": game_ids}
            df, sample_meta["top_score"] = _get_slate_df(
                session, game_ids, model_names, cache_dir, cache_mode
            )

            df.to_parquet(
                os.path.join(dataset_dest_dir, f"{season}-{game_number}-{game_ids_hashed}.pq")
            )

            samples_meta.append(sample_meta)

    with open(os.path.join(dataset_dest_dir, "samples_meta.json"), "w") as f_:
        json.dump(
            {
                "sport": db_obj.db_manager.ABBR,
                "seasons": seasons,
                "seed": seed,
                "samples": samples_meta,
            },
            f_,
            indent="\t",
        )


def main(cmd_line_str=None):
    parser = argparse.ArgumentParser(
        description="Functions to export data for, train and test deep learning lineup generator models"
    )

    parser.add_argument("--cache_dir", default=None, help="Folder to cache to")
    parser.add_argument("--cache_mode", choices=CacheMode.__args__)

    parser.add_argument("--seed", default=0)
    parser.add_argument(
        "--samples",
        "--cases",
        "--n",
        type=int,
        help=f"How many samples to infer. default={_DEFAULT_SAMPLE_COUNT}",
        default=_DEFAULT_SAMPLE_COUNT,
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        help="Default is to infer slates from any available season",
    )
    parser.add_argument(
        "--games_per_slate_range",
        help=f"Number of games per slate. Default is {_DEFAULT_GAMES_PER_SLATE}",
        nargs=2,
        type=int,
        default=_DEFAULT_GAMES_PER_SLATE,
    )
    parser.add_argument(
        "--dest_dir",
        help="Directory under which dataset directories will be written. "
        f"Default is '{_DEFAULT_PARENT_DATASET_PATH}'",
        default=_DEFAULT_PARENT_DATASET_PATH,
    )
    parser.add_argument(
        "--existing_files_mode",
        "--file_mode",
        choices=_ExistingFilesMode.__args__,
        default="append",
        help="Default is append, new files will overwrite or add to existing files in the dataset directory",
    )
    parser.add_argument("--name", help="Name of the dataset. Default is a datetime stamp")
    parser.add_argument("db_file")

    service_names = CLSRegistry.get_names(FANTASY_SERVICE_DOMAIN)
    service_abbrs = {
        CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, name).ABBR.lower(): name
        for name in service_names
    }
    parser.add_argument(
        "service", choices=sorted(service_names + tuple(service_abbrs.keys())), required=True
    )

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    db_obj = db.get_db_obj(args.db_file)
    service_class = CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, args.service)
    model_names = getattr(service_class, "DEFAULT_MODEL_NAMES", {}).get(db_obj.db_manager.ABBR)
    _export_deep_dataset(
        db_obj,
        args.name,
        args.dest_dir,
        args.seasons,
        args.games_per_slate_range,
        args.samples,
        args.existing_files_mode,
        model_names,
        args.seed,
        args.cache_dir,
        args.cache_mode if args.cache_dir else "disabled",
    )


if __name__ == "__main__":
    main()
