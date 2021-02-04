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
    def get_entries_df(cls, history_file_dir):
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
        entries_df.Sport = entries_df.Sport.str.lower()
        entries_df["Date"] = pd.to_datetime(entries_df.Contest_Date_EST)
        return entries_df

    def process_entry(self, entry_info):
        # go to contest page
        link = f"https://www.draftkings.com/contest/gamecenter/{entry_info.Contest_Key}?uc={entry_info.Entry_Key}#/"
        self.browse_to(link)

        # get draft % for all players in my lineup
        my_lineup_element = self.browser.find_elements_by_xpath(
            '//label[@id="multiplayer-live-scoring-Rank"]/../../../../div'
        )[1]
        raise NotImplementedError("use the text from this as a direct input to a dataframe")

        # if contest has been processed then we are done
        if contest_info.Contest_Key in self.processed_contests:
            return
        raise NotImplementedError()

        # if contest has been processed then we are done

        # get top score

        # get last winning score

        # get draft % for all players in top 5 lineups

        # get draft % for last winning lineup
