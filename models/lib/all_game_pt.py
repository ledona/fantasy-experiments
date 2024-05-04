"""
Predict results for multiple players/teams in a game in one shot. Input sample
is historic stats on the players and teams that will be playing in the game.
Target is a vector with target stats for game players/teams for the game
"""

import argparse
import json
import os
import shlex
from typing import Literal

from fantasy_py import JSONWithCommentsDecoder, log

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

    @property
    def xform_src_files(self) -> dict[Literal["team", "player"], list[str]]:
        """returns the src data files"""
        return self._def_dict["data_filenames"]


def _data_func(args, model_def: _ModelDefinitionFile):
    if not os.path.isdir(args.dest_path):
        args.parser.error(f"Dest path '{args.dest_path}' is not a directory.")
    dest_filepath = os.path.join(args.dest_path, model_def.model_name + ".pq")
    if not os.path.isdir(args.input_data_dir):
        args.parser.error(f"Input data path '{args.input_data_dir}' does not exist")

    raise NotImplementedError("validate src files, this is a dict of lists of filepaths")
    src_filepaths = os.path.join(args.input_data_dir, model_def.xform_src_files)
    _LOGGER.info(
        "Generating training data for model='%s'. dest-file='%s', src-files=%s",
        model_def.model_name,
        dest_filepath,
        src_filepaths,
    )
    raise NotImplementedError()


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
