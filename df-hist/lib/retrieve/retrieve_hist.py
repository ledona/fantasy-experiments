import logging
import os
import shlex
from argparse import ArgumentParser
from datetime import timedelta

import pandas as pd
from dateutil.parser import parse as du_parse
from tqdm import tqdm

from . import log
from .service_data_retriever import (
    EXPECTED_HISTORIC_ENTRIES_DF_COLS,
    DataUnavailable,
    NavigationAvailableError,
    ServiceDataRetCacheMode,
    ServiceDataRetriever,
    WebLimitReached,
    get_service_data_retriever,
)

_LOGGER = logging.getLogger(__name__)

_ACCEPTED_SPORTS = {"nhl", "nfl", "mlb", "nba", "lol"}


def retrieve_history(
    service_name,
    history_file_dir,
    sports=None,
    start_date=None,
    end_date=None,
    cache_path=None,
    cache_mode: ServiceDataRetCacheMode | None = None,
    cache_only=False,
    interactive=False,
    web_limit=None,
    browser_debug_address=None,
    browser_debug_port=None,
    profile_path=None,
    skip_entry_filepath=None,
) -> tuple[ServiceDataRetriever, int]:
    """
    sports: collection of sports to process, if None then process all sports
    browser_debug_address: address of an existing chrome browser to use, ip-address:port
    browser_debug_port: port to request that the new browser used for debugging \
        will be available on
    interactive: require user response prior to every browser action

    returns ServiceDataRetriever, entry_count
    """
    assert (browser_debug_address is None) or (browser_debug_port is None)

    if skip_entry_filepath:
        _LOGGER.info("Loading skip entry data from '%s'", skip_entry_filepath)
        with open(skip_entry_filepath, "r") as sef:
            skip_entries = set()
            for i, line in enumerate(sef.readlines(), 1):
                entry = line.strip().split(":")
                if len(entry) != 3:
                    raise ValueError(
                        f"Failed to load entries from '{skip_entry_filepath}'. {len(entry)} values found on line {i}."
                    )
                sport, date_str, title = entry
                if title[0] == "'":
                    title = title[1:]
                if title[-1] == "'":
                    title = title[:-1]
                if sport not in _ACCEPTED_SPORTS:
                    raise ValueError(
                        f"Failed to load entries from '{skip_entry_filepath}'. on line {i}, {sport=} is not valid"
                    )
                try:
                    date_ = du_parse(date_str).date()
                except Exception as ex:
                    raise ValueError(
                        f"Failed to load entries from '{skip_entry_filepath}'. on line {i}, failed to parse {date_str=}"
                    ) from ex

                skip_entries.add((sport, date_, title))
    else:
        skip_entries = None

    service_obj = get_service_data_retriever(
        service_name,
        cache_path=cache_path,
        cache_mode=cache_mode,
        cache_only=cache_only,
        browser_profile_path=profile_path,
        browser_address=browser_debug_address,
        browser_debug_port=browser_debug_port,
        interactive=interactive,
        web_limit=web_limit,
    )
    # do this first since
    assert (sports is None) or set(sports) == {sport.lower() for sport in sports}, (
        "all sports must be in lower case"
    )
    contest_entries_df = service_obj.get_historic_entries_df_from_file(history_file_dir)
    assert (
        len(missing_cols := EXPECTED_HISTORIC_ENTRIES_DF_COLS - set(contest_entries_df.columns))
        == 0
    ), f"dataframe does not have the following required columns: {missing_cols}"
    entry_count = len(contest_entries_df)

    filters = ["entries > 0"]
    if start_date is not None:
        filters.append("date >= @start_date")
    if end_date is not None:
        filters.append("date <= @end_date")

    sports_filter = sports or _ACCEPTED_SPORTS
    filters.append("sport in @sports_filter")

    if len(filters) > 0:
        contest_entries_df = contest_entries_df.query(" and ".join(filters))
        if (removed_entries := entry_count - len(contest_entries_df)) > 0:
            _LOGGER.info("%i rows filtered out", removed_entries)
            entry_count -= removed_entries
    _LOGGER.info("%i entries to process", entry_count)
    if entry_count == 0:
        raise ValueError("No entries to process!")

    with tqdm(total=entry_count, desc="entries") as pbar:
        pbar.set_postfix(cache=0, web=0)

        def func(entry_info):
            pbar.update()
            pbar.set_postfix(
                {**service_obj.processed_counts_by_src, "date": entry_info.date.date()}
            )

            if (
                skip_entries
                and (skip_entry_key := (entry_info.sport, entry_info.date.date(), entry_info.title))
                in skip_entries
            ):
                _LOGGER.info("*** SKIPPING ENTRY %s ***", skip_entry_key)
                service_obj.processed_counts_by_src["skipped"] += 1
                result = None
            else:
                # this will be true if currently retrying failed navigation
                currently_retrying_nav = False

                while True:
                    try:
                        result = service_obj.process_entry(entry_info)
                    except DataUnavailable:
                        _LOGGER.warning(
                            "Skipping entry missing from cache: %s:%s:'%s'",
                            entry_info.sport,
                            entry_info.date.strftime("%Y%m%d"),
                            entry_info.title,
                        )
                        service_obj.processed_counts_by_src["skipped"] += 1
                        result = None
                    except NavigationAvailableError:
                        _LOGGER.warning(
                            "*** Encountered a navigation error for %s:%s:'%s', %s ***",
                            entry_info.sport,
                            entry_info.date.strftime("%Y%m%d"),
                            entry_info.title,
                            "will retry" if not currently_retrying_nav else "failed twice",
                        )
                        if currently_retrying_nav:
                            raise
                        currently_retrying_nav = True
                        continue

                    break
            pbar.set_postfix(
                {**service_obj.processed_counts_by_src, "date": entry_info.date.date()}
            )
            return result

        try:
            contest_entries_df.sort_values(["date", "title"]).apply(func, axis=1)
        except WebLimitReached as limit_reached_ex:
            _LOGGER.info(
                "Web retrieval limit was reached before retrieval attempt for %s",
                limit_reached_ex.args[0],
            )
        finally:
            _LOGGER.info(
                "Entry processing done. %i entries processed. Entries processed by data source: %s",
                sum(service_obj.processed_counts_by_src.values()),
                service_obj.processed_counts_by_src,
            )

    return service_obj, entry_count


def process_cmd_line(cmd_line_str=None):
    parser = ArgumentParser(
        description="Retrieve historic contest history from daily fantasy services"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Prompt user to continue prior to all browser actions",
    )
    parser.add_argument("--cache-path", "--cache", help="path to cached files")

    parser.add_argument(
        "--skip-entry-filepath",
        help="Path to a text file containing entries to skip. "
        "Each line should be of the form {sport}:{date}:{entry title}",
    )

    mut_ex_group = parser.add_mutually_exclusive_group()
    mut_ex_group.add_argument("--cache-mode", choices=ServiceDataRetCacheMode.__args__)
    mut_ex_group.add_argument("--cache-only", action="store_true", default=False)

    mut_ex_group = parser.add_mutually_exclusive_group()
    mut_ex_group.add_argument(
        "--chrome-debug-port",
        "--chrome-port",
        "--port",
        help="Create a chrome instance and make this the debug port",
    )
    mut_ex_group.add_argument(
        "--chrome-debug-address",
        "--chrome-address",
        "--address",
        help="Address of chrome instance to connect to",
    )
    parser.add_argument(
        "--chrome-profile-path",
        "--profile-path",
        help="path to chrome user profile, only valid if --chrome-debug-address is NOT used",
    )
    parser.add_argument(
        "--history-file-dir",
        default=".",
        help="Path to directory containing downloaded contest history files. Default='./'",
    )
    parser.add_argument(
        "--write",
        "-o",
        const=".",
        nargs="?",
        metavar="DIR",
        help="Write the results to this directory. Default is the current directory. "
        "Results will be written to files with names {service}.[betting|contest|draft].csv",
    )
    parser.add_argument(
        "--sports",
        help="Sports to process",
        nargs="+",
        choices=_ACCEPTED_SPORTS,
    )
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", help="End date (inclusive) in YYYY-MM-DD format")
    parser.add_argument(
        "--web-limit",
        type=int,
        default=3,
        help="Only processes this number of entries retrieved from internet. Default is 3",
    )
    parser.add_argument("service", choices=("fanduel", "draftkings", "yahoo"))

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    if (args.chrome_debug_address is not None) and (args.chrome_profile_path is not None):
        parser.error("--chrome-profile-path cannot be used with --chrome-debug-address")
    if (args.cache_only is True) and (args.cache_path is None):
        parser.error("--cache_path is required if --cache_only is used")
    _LOGGER.info("starting data retrieval")
    start_date = du_parse(args.start_date).date() if args.start_date is not None else None
    end_date = (
        (du_parse(args.end_date) + timedelta(days=1)).date() if args.end_date is not None else None
    )
    service_obj, entry_count = retrieve_history(
        args.service,
        args.history_file_dir,
        sports=args.sports,
        start_date=start_date,
        end_date=end_date,
        browser_debug_port=args.chrome_debug_port,
        browser_debug_address=args.chrome_debug_address,
        profile_path=args.chrome_profile_path,
        cache_path=args.cache_path,
        cache_mode=args.cache_mode,
        cache_only=args.cache_only,
        interactive=args.interactive,
        web_limit=args.web_limit,
        skip_entry_filepath=args.skip_entry_filepath,
    )

    if len(service_obj.processed_contests) == 0:
        _LOGGER.warning("Nothing was processed!")
    else:
        if args.write is not None:
            _LOGGER.info("Writing CSV files to '%s'", args.write)
            service_obj.contest_df.to_csv(
                os.path.join(args.write, args.service + ".contest.csv"), index=False
            )
            service_obj.player_draft_df.to_csv(
                os.path.join(args.write, args.service + ".draft.csv"), index=False
            )
            service_obj.entry_df.to_csv(
                os.path.join(args.write, args.service + ".betting.csv"), index=False
            )
        else:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.expand_frame_repr",
                False,
            ):
                print("\nContest History")
                print(service_obj.contest_df)
                print("\nBetting History")
                print(service_obj.entry_df)
                print("\nDraft History")
                print(service_obj.player_draft_df)

    _LOGGER.info(
        "Done! %i / %i entries processed. %s",
        sum(service_obj.processed_counts_by_src.values()),
        entry_count,
        service_obj.processed_counts_by_src,
    )


if __name__ == "__main__":
    log.setup()
    process_cmd_line()
