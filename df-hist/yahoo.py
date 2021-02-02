import glob
import logging
import os

import pandas as pd
from selenium.webdriver.common.by import By

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Yahoo(ServiceDataRetriever):
    SERVICE_URL = "https://sports.yahoo.com/dailyfantasy/"
    LOC_SIGN_IN = (By.LINK_TEXT, "Sign in")
    LOC_LOGGED_IN = (By.NAME, "Profile Picture")

    def get_entries_df(self, history_file_dir):
        """ return an iterator that yields entries """
        # get the most recent dk contest entry filename
        glob_pattern = os.path.join(history_file_dir, "Yahoo_DF_my_contest_history.*.csv")
        glob_pattern = os.path.expanduser(glob_pattern)
        history_filenames = glob.glob(glob_pattern)

        if len(history_filenames) == 0:
            raise FileNotFoundError(f"No history files found for '{glob_pattern}'")
        history_filename = sorted(history_filenames)[-1]
        LOGGER.info("Loading history data from '%s'", history_filename)
        entries_df = pd.read_csv(history_filename)
        entries_df["Date"] = pd.to_datetime(entries_df['Start Date'])
        entries_df.Sport = entries_df.Sport.str.lower()

        return entries_df

    def process_entry(self, entry_info):
        """
        process a contest entry. if the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        raise NotImplementedError()
