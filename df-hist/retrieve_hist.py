from argparse import ArgumentParser
import logging
import shlex

import pandas as pd

from service_data_retriever import get_service_data_retriever


LOGGING_FORMAT = '%(asctime)s-%(levelname)s-%(name)s(%(lineno)s)-%(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

LOGGER = logging.getLogger(__name__)
DEFAULT_HISTORY_FILE_DIR = "~/Google Drive/fantasy"


def retrieve_history(service_name, username, password, history_file_dir,
                     sport=None, start_date=None, end_date=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    returns (contest history dataframe, player draft history dataframe, betting dataframe)
    """
    service_obj = get_service_data_retriever(service_name)
    service_obj.login(username, password)

    for entry_info in service_obj.get_entries(history_file_dir, sport, start_date, end_date):
        service_obj.process_entry(entry_info)

    return service_obj.contest_history_df, service_obj.draft_history_df, service_obj.betting_history_df


def process_cmd_line(cmd_line_str=None):
    parser = ArgumentParser(description="Retrieve historic contest history from daily fantasy services")

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
    parser.add_argument("--sport")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "service", choices=("fanduel", "draftkings", "yahoo")
    )
    parser.add_argument("username")
    parser.add_argument("password")

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    LOGGER.info("starting data retrieval")

    contest_history_df, player_draft_df, betting_history_df = retrieve_history(
        args.service, args.username, args.password,
        args.history_file_dir,
        sport=args.sport,
        start_date=args.start_date,
        end_date=args.end_date,
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
