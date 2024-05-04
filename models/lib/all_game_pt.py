"""
Predict results for multiple players/teams in a game in one shot. Input sample
is historic stats on the players and teams that will be playing in the game.
Target is a vector with target stats for game players/teams for the game
"""

import argparse
import functools
import json
import os
import shlex

import pandas as pd
from fantasy_py import JSONWithCommentsDecoder, StatInfo, UnexpectedValueError, log

from .train_test import ArchitectureType

_DEFAULT_ARCHITECTURE: ArchitectureType = "tpot"

_LOGGER = log.get_logger(__name__)


class _ModelDefinitionFile:
    """object that parses a model definition json file"""

    def __init__(self, filepath: str, model_name: str):
        with open(filepath, "r") as f_:
            self._def_dict = json.load(f_, cls=JSONWithCommentsDecoder)

        self.model_name = model_name
        """name of the model"""

    @functools.lru_cache
    def get_xform_src_filepaths(self, dir_path):
        """returns the src data filepaths"""
        filepaths: list[str] = []
        expected_games: set[int] | None = None
        expected_seasons = set(self.training_seasons + [self.test_season])

        for filename in self._def_dict["datafiles"]:
            filepath = os.path.join(dir_path, filename)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(
                    f"Input data file '{filename}' not found in data path '{dir_path}'"
                )
            filepaths.append(filepath)

            df = pd.read_parquet(filepath)
            if missing_seasons := len(expected_seasons - set(df.seasons.unique())) > 0:
                raise UnexpectedValueError(
                    f"The following required seasons are missing from  '{filename}': {missing_seasons} "
                )
            if expected_games is None:
                expected_games = set(df.game_id.unique())
            elif not set(df.game_id.unique()) == expected_games:
                raise UnexpectedValueError(
                    "the games in input file '{filename}' do not match other input file games!"
                )

        return filepaths

    @functools.cached_property
    def features(self) -> list[StatInfo]:
        """historic and game descriptive stats"""
        raise NotImplementedError()

    @functools.cached_property
    def targets(self) -> list[StatInfo]:
        """values to predict"""
        raise NotImplementedError()

    @functools.cached_property
    def training_seasons(self):
        if "training_seasons" not in self._def_dict:
            raise UnexpectedValueError("training seasons not found")
        train_seasons: list[int] = []
        for season in self._def_dict["training_seasons"]:
            if not isinstance(season, int):
                raise UnexpectedValueError("Error parsing training seasons")
            train_seasons.append(season)
        return train_seasons

    @functools.cached_property
    def test_season(self):
        season = self._def_dict["validation_season"]
        if not isinstance(season, int):
            raise UnexpectedValueError("Error parsing test/validation season")
        return season


def _create_dataset(
    dest_filepath: str,
    seasons: list[int],
    features: list[StatInfo],
    targets: list[StatInfo],
    data_filepaths: list[str],
):
    _LOGGER.info("Creating dataset at '%s' for seasons=%s", dest_filepath, seasons)

    raise NotImplementedError()


def _data_func(args, model_def: _ModelDefinitionFile):
    if not os.path.isdir(args.dest_path):
        args.parser.error(f"Dest path '{args.dest_path}' is not a directory.")
    dest_filepath = os.path.join(args.dest_path, model_def.model_name)
    if not os.path.isdir(args.input_data_dir):
        args.parser.error(f"Input data path '{args.input_data_dir}' does not exist")

    src_filepaths = model_def.get_xform_src_filepaths(args.input_data_dir)

    _LOGGER.info(
        "Generating training data for model='%s'. dest-files='%s-[train|test].pq', src-files=%s",
        model_def.model_name,
        dest_filepath,
        src_filepaths,
    )
    _LOGGER.info("model '%s' targets=%s", model_def.model_name, model_def.targets)
    _LOGGER.info("model '%s' input-features=%s", model_def.model_name, model_def.features)

    _create_dataset(
        args.dest_filepath + "-train.pq",
        model_def.training_seasons,
        model_def.features,
        model_def.targets,
        src_filepaths,
    )
    _create_dataset(
        args.dest_filepath + "-train.pq",
        model_def.test_season,
        model_def.features,
        model_def.targets,
    )


def _add_data_parser(sub_parsers):
    parser = sub_parsers.add_parser("data", help="Transform a data export to model training data")
    parser.set_defaults(func=_data_func, parser=parser)
    parser.add_argument(
        "input_data_dir",
        help="parent directory for input data that will be transformed to training data",
    )
    parser.add_argument(
        "dest_path",
        help="Path to write training dataset. If this ends with the extention '.pq' "
        "then write output to this filepath. Otherwise treat this as a folder and "
        "create a file here with the trainging data.",
    )


def _train_func(args, model_def: _ModelDefinitionFile):
    raise NotImplementedError()


def _add_train_parser(sub_parsers):
    parser = sub_parsers.add_parser("train", help="Train the model")
    parser.set_defaults(func=_train_func, parser=parser)
    parser.add_argument(
        "--arch",
        default=_DEFAULT_ARCHITECTURE,
        choices=ArchitectureType.__args__,
        help="model architecture",
    )


def main(cmd_line_str=None):
    log.set_default_log_level(only_fantasy=False)
    log.enable_progress()

    parser = argparse.ArgumentParser(
        description="Functions to export data and train models for game wide stats"
    )

    parser.add_argument("cfg_file", help="the configuration json file defining the model")
    parser.add_argument("model", help="name of the model in the configuration file")
    subparsers = parser.add_subparsers()
    _add_data_parser(subparsers)

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)
    if not hasattr(args, "func"):
        parser.print_help()
        parser.exit(1)

    model_def = _ModelDefinitionFile(args.cfg_file, args.model)
    args.func(args, model_def)


if __name__ == "__main__":
    main()
