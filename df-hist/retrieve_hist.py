from argparse import ArgumentParser
import logging
import shlex

import pandas as pd
from tqdm import tqdm

from service_data_retriever import get_service_data_retriever, EXPECTED_ENTRIES_COLS


LOGGING_FORMAT = '%(asctime)s-%(levelname)s-%(name)s(%(lineno)s)-%(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOGGER = logging.getLogger(__name__)
DEFAULT_HISTORY_FILE_DIR = "~/Google Drive/fantasy/betting"


def retrieve_history(
        service_name, history_file_dir,
        sports=None, start_date=None, end_date=None,
        cache_path=None, interactive=False,
        browser_debug_address=None, browser_debug_port=None, profile_path=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    sports - collection of sports to process, if None then process all sports
    browser_debug_address - address of an existing chrome browser to use, ip-address:port
    browser_debug_port - port to request that the new browser used for debugging will be available on
    interactive - require user response prior to every browser action

    returns (contest history dataframe, player draft history dataframe, betting dataframe)
    """
    assert (browser_debug_address is None) or (browser_debug_port is None)
    service_obj = get_service_data_retriever(service_name,
                                             cache_path=cache_path,
                                             browser_profile_path=profile_path,
                                             browser_address=browser_debug_address,
                                             browser_debug_port=browser_debug_port,
                                             interactive=interactive)
    # do this first since
    assert (sports is None) or set(sports) == {sport.lower() for sport in sports}, \
        "all sports must be in lower case"
    contest_entries_df = service_obj.get_entries_df(history_file_dir)
    assert EXPECTED_ENTRIES_COLS <= set(contest_entries_df.columns), \
        f"dataframe does not have the following required columns: {EXPECTED_ENTRIES_COLS - set(contest_entries_df.columns)}"
    entry_count = len(contest_entries_df)

    filters = []
    if start_date is not None:
        filters.append("date >= @start_date")
    if end_date is not None:
        filters.append("date <= @end_date")
    if sports is not None:
        filters.append("sport in @sports")

    if len(filters) > 0:
        contest_entries_df = contest_entries_df.query(" and ".join(filters))
        if (removed_entries := entry_count - len(contest_entries_df)) > 0:
            LOGGER.info("%i rows filtered out", removed_entries)
            entry_count -= removed_entries
    LOGGER.info("%i entries to process", entry_count)
    if entry_count == 0:
        raise ValueError("No entries to process!")

    tqdm.pandas(desc="entries")
    contest_entries_df.progress_apply(service_obj.process_entry, axis=1)

    return service_obj.contest_df, service_obj.player_draft_df, service_obj.entry_df


def process_cmd_line(cmd_line_str=None):
    parser = ArgumentParser(description="Retrieve historic contest history from daily fantasy services")

    parser.add_argument(
        "--interactive", action="store_true", default=False,
        help="Prompt user to continue prior to all browser actions"
    )
    parser.add_argument("--cache-path", "--cache", help="path to cached files")
    mut_ex_group = parser.add_mutually_exclusive_group()
    mut_ex_group.add_argument(
        "--chrome-debug-port", "--chrome-port", "--port",
        help="Debug chrome port request be made available on the created chrome instance"
    )
    mut_ex_group.add_argument("--chrome-debug-address", "--chrome-address", "--address",
                              help="Address of chrome instance to connect to")
    parser.add_argument(
        "--chrome-profile-path", "--profile-path",
        help="path to chrome user profile, only valid if --chrome-debug-address is NOT used"
    )
    parser.add_argument(
        "--history-file-dir", default=DEFAULT_HISTORY_FILE_DIR,
        help=(
            "Path to directory containing downloaded contest history files. "
            f"Default={DEFAULT_HISTORY_FILE_DIR}"
        )
    )
    parser.add_argument(
        "--filename-prefix", "-o", metavar="filename-prefix",
        help=("Output filename prefix. Create 2 csv files containing results, "
              "one for contest history the other for player draft history). "
              "Filenames will be prefixed with this value.")
    )
    parser.add_argument("--sports", help="Sports to process", nargs="+",
                        choices=('nhl', 'nfl', 'mlb', 'nba', 'lol'))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "service", choices=("fanduel", "draftkings", "yahoo")
    )

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    if (args.chrome_debug_address is not None) and (args.chrome_profile_path is not None):
        parser.error("--chrome-profile-path cannot be used with --chrome-debug-address")
    LOGGER.info("starting data retrieval")

    contest_history_df, player_draft_df, betting_history_df = retrieve_history(
        args.service,
        args.history_file_dir,
        sports=args.sports,
        start_date=args.start_date,
        end_date=args.end_date,
        browser_debug_port=args.chrome_debug_port,
        browser_debug_address=args.chrome_debug_address,
        profile_path=args.chrome_profile_path,
        cache_path=args.cache_path,
        interactive=args.interactive,
    )

    if args.filename_prefix:
        LOGGER.info("Writing CSV files")
        contest_history_df.to_csv(args.filename_prefix + ".contest.csv")
        player_draft_df.to_csv(args.filename_prefix + ".draft.csv")
        betting_history_df.to_csv(args.filename_prefix + ".betting.csv")
    else:
        print(contest_history_df)
        print(player_draft_df)
        print(betting_history_df)

    LOGGER.info("done")


if __name__ == "__main__":
    logging.basicConfig(format=LOGGING_FORMAT, datefmt=DATE_FORMAT, level=logging.INFO)
    process_cmd_line()
