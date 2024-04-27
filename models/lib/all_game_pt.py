"""
Predict results for multiple players/teams in a game in one shot. Input sample
is historic stats on the players and teams that will be playing in the game.
Target is a vector with target stats for game players/teams for the game
"""

import argparse
import shlex

from fantasy_py import log

from .train_test import ArchitectureType

_DEFAULT_ARCHITECTURE: ArchitectureType = "tpot"


def _train_func(args):
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


def _data_func(args):
    raise NotImplementedError()


def _add_data_parser(sub_parsers):
    parser = sub_parsers.add_parser("data", help="Transform a data export to model training data")
    parser.set_defaults(func=_data_func, parser=parser)


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
    args.func(args)


if __name__ == "__main__":
    main()
