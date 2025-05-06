import os
import shlex
from argparse import ArgumentParser
from functools import partial

import dateutil
import pandas as pd
from fantasy_py import CONTEST_DOMAIN, CacheSettings, CLSRegistry, DFSContestStyle, log
from fantasy_py.betting import FiftyFifty, GeneralPrizePool
from tqdm import tqdm

from ..data_cfg import SPORT_CFGS
from .best_possible_lineup_score import TopScoreCacheMode
from .data_xform import xform

_DEFAULT_TOP_PERCENTILE = 0.7
_DEFAULT_DATA_PARENT_DIR = "/fantasy-isync/fantasy-dfs-hist"
_DEFAULT_INPUT_DATA_DIR = os.path.join(_DEFAULT_DATA_PARENT_DIR, "betting")
_DEFAULT_OUTPUT_DATA_DIR = os.path.join(_DEFAULT_DATA_PARENT_DIR, "data")


def _process_cmd_line(cmd_line_str=None):
    parser = ArgumentParser(
        description="Transform data from raw historic DFS betting history to "
        "training data for min/max expected winning scores"
    )

    parser.add_argument(
        "--top_percentile",
        help="players/teams above this percentile are considered top performers",
        default=_DEFAULT_TOP_PERCENTILE,
    )

    parser.add_argument(
        "--services",
        help="default='draftkings'",
        nargs="+",
        choices=["draftkings", "fanduel", "yahoo"],
        default=["draftkings"],
    )

    parser.add_argument(
        "--contest_styles",
        "--styles",
        nargs="+",
        type=DFSContestStyle,
        choices=[DFSContestStyle.CLASSIC.name, DFSContestStyle.SHOWDOWN.name],
        default=[DFSContestStyle.CLASSIC, DFSContestStyle.SHOWDOWN],
    )

    parser.add_argument(
        "--contest_types",
        "--types",
        nargs="+",
        type=partial(CLSRegistry, CONTEST_DOMAIN),
        choices=[FiftyFifty.NAME, GeneralPrizePool.NAME],
        default=[FiftyFifty, GeneralPrizePool],
    )

    parser.add_argument(
        "--top_score_cache_mode", choices=TopScoreCacheMode.__args__, default="default"
    )

    parser.add_argument(
        "--data_path",
        default=_DEFAULT_OUTPUT_DATA_DIR,
        help=f"Directory where data will be written.default='{_DEFAULT_OUTPUT_DATA_DIR}'",
    )

    parser.add_argument(
        "--no_df",
        dest="show_df",
        default=True,
        action="store_false",
        help="Do not print the resulting data to stdout on completion. "
        "(default is to show the data)",
    )
    parser.add_argument(
        "--contest_data_path",
        default=_DEFAULT_INPUT_DATA_DIR,
        help=f"Directory containing contest data. default='{_DEFAULT_INPUT_DATA_DIR}'",
    )

    parser.add_argument(
        "--date_range",
        "--dates",
        nargs=2,
        metavar=("start-date", "end-date"),
        type=lambda date_str: dateutil.parser.parse(date_str).date(),
        help="Start and end date to process. Start is inclusive, end exclusive. "
        "Default is the dates in the config for the sport",
    )

    parser.add_argument("--sports", nargs="+", choices=SPORT_CFGS.keys(), default=SPORT_CFGS.keys())
    CacheSettings.add_parser_args(parser, dir_create_mode_default="prompt")

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    print(f"{args=}")

    dfs: dict[tuple, pd.DataFrame] = {}
    for sport in (tqdm_progress := tqdm(sorted(set(args.sports)), desc="sport")):
        tqdm_progress.set_postfix_str(sport)
        dfs.update(
            xform(
                sport,
                SPORT_CFGS[sport],
                sorted(set(args.services)),
                sorted(set(args.contest_styles)),
                sorted(set(args.contest_types), key=lambda ct: ct.NAME),
                args.top_score_cache_mode,
                args.data_path,
                args.contest_data_path,
                args.top_percentile,
                args.date_range,
            )
        )

    if args.show_df:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
            "expand_frame_repr",
            False,
        ):
            for desc, df in dfs.items():
                print(f"data descriptor: {desc}")
                print(df)


if __name__ == "__main__":
    log.configure_logging(progress=True)
    _process_cmd_line()
