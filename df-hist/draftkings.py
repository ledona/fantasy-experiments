import glob
import logging
import os

import pandas as pd
from selenium.webdriver.common.by import By

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Draftkings(ServiceDataRetriever):
    SERVICE_URL = "https://www.draftkings.com"
    LOC_SIGN_IN = (By.LINK_TEXT, "Sign In")
    LOC_LOGGED_IN = (By.LINK_TEXT, "Lobby")

    @classmethod
    def get_entries(cls, history_file_dir, sport, start_date, end_date):
        """ return an iterator that yields contest entries """
        # get the most recent dk contest entry filename
        glob_pattern = os.path.join(history_file_dir, "draftkings-contest-entry-history.*.csv")
        glob_pattern = os.path.expanduser(glob_pattern)
        history_filenames = glob.glob(glob_pattern)

        if len(history_filenames) == 0:
            raise FileNotFoundError(f"No history files found for '{glob_pattern}'")
        history_filename = sorted(history_filenames)[-1]
        LOGGER.info("Loading history data from '%s'", history_filename)
        entries_df = pd.read_csv(history_filename)

        raise NotImplementedError()

    def process_entry(self, entry_info):
        """
        process a contest entry. if the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        raise NotImplementedError()
