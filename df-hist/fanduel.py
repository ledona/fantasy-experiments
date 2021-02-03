from collections import defaultdict
import glob
import logging
import os

import pandas as pd
from selenium.webdriver.common.by import By

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Fanduel(ServiceDataRetriever):
    SERVICE_URL = "https://www.fanduel.com"
    LOC_SIGN_IN = (By.LINK_TEXT, "Log in")
    LOC_LOGGED_IN = (By.LINK_TEXT, "Lobby")

    def get_entries_df(self, history_file_dir):
        glob_pattern = os.path.join(history_file_dir, "fanduel entry history *.csv")
        glob_pattern = os.path.expanduser(glob_pattern)
        history_filenames = glob.glob(glob_pattern)

        if len(history_filenames) == 0:
            raise FileNotFoundError(f"No history files found for '{glob_pattern}'")

        # find the most recent date
        retrieval_date_filenames = defaultdict(list)
        for filename in history_filenames:
            retrieval_date_filenames[filename.rsplit(' ', 1)[1][:8]].append(filename)
        most_recent_date = sorted(retrieval_date_filenames.keys())[-1]
        LOGGER.info("Loading history data from '%s'", retrieval_date_filenames[most_recent_date])
        dfs = (
            pd.read_csv(filename, index_col=False)
            for filename in sorted(retrieval_date_filenames[most_recent_date])
        )
        entries_df = pd.concat(dfs)
        rows_of_data = len(entries_df)

        # convert dates and drop rows with invalid dates (happens for cancelled contests)
        entries_df.Date = pd.to_datetime(entries_df.Date, errors='coerce')
        entries_df = entries_df[entries_df.Date.notna()]
        if (invalid_dates := rows_of_data - len(entries_df)) > 0:
            LOGGER.info("%i invalid dates found. dropping those entries", invalid_dates)
            rows_of_data = len(entries_df)

        return entries_df

    def process_entry(self, entry_info):
        """
        process a contest entry. if the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        # go to page in entry_info.Link
        self.browse_to(entry_info.Link)

        # get draft % for all players in my lineup
        raise NotImplementedError()

        # if contest has been processed then we are done

        # get top score

        # get last winning score

        # get draft % for all players in top 5 lineups

        # get draft % for last winning lineup
