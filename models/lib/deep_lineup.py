"""functions required for exporting and training deep models"""
import argparse
import json
import os
import shlex
import shutil
from random import Random
from typing import Literal, cast

from fantasy_py import FantasyException, db, log, dt_to_filename_str
from tqdm import tqdm

_DEFAULT_SAMPLE_COUNT = 10
_DEFAULT_PARENT_PATH = os.path.join(".", "deep-lineup-datasets")
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


def _export_deep_dataset(
    db_file: str,
    dataset_name,
    parent_dest_dir: str,
    requested_seasons: list[int] | None,
    case_count: int,
    existing_files_mode: _ExistingFilesMode,
    seed,
):
    _LOGGER.info("Starting export")
    dataset_dest_dir = _prep_dataset_directory(parent_dest_dir, dataset_name, existing_files_mode)
    db_obj = db.get_db_obj(db_file)
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
    rand_season = Random(seed)
    for _ in (prog_iter := tqdm(range(case_count))):
        season = rand_season.choice(seasons)
        game_num = rand_season.randint(1, db_obj.db_manager.get_max_epochs(season))
        prog_iter.set_postfix_str(f"{season}-{game_num}")
        sample_meta = {"season": season, "game_num": game_num}

        raise NotImplementedError()

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
        "--dest_dir",
        help=f"Directory under which dataset directories will be written. Default is '{_DEFAULT_PARENT_PATH}'",
        default=_DEFAULT_PARENT_PATH,
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

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    _export_deep_dataset(
        args.db_file,
        args.name,
        args.dest_dir,
        args.seasons,
        args.samples,
        args.existing_files_mode,
        args.seed,
    )


if __name__ == "__main__":
    main()
