from collections import defaultdict
import glob
import logging
import os

import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Fanduel(ServiceDataRetriever):
    SERVICE_URL = "https://www.fanduel.com"
    LOC_SIGN_IN = (By.LINK_TEXT, "Log in")
    LOC_LOGGED_IN = (By.LINK_TEXT, "Lobby")
    
    # use longer waits for captcha
    LOGIN_TIMEOUT = 300
    WAIT_TIMEOUT = 300

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

    def get_entry_lineup_data(self, link, title):
        self.browse_to(link, title=title)

        # get draft % for all players in my lineup
        my_lineup_element = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//div[@data-test-id="contest-entry"]')),
            "Waiting for lineup"
        )

        return my_lineup_element.text

    def get_entry_lineup_df(self, lineup_data):
        raise NotImplementedError()

    def get_contest_data(self, link, title) -> dict:
        self.browse_to(link, title=title)

        min_winning_score = self.browser.find_element_by_xpath('//span[@data-test-id="RunningManScore"]').text
        winning_score = self.browser.find_element_by_xpath('//table[@data-test-id="contest-entry-table"]//tbody/tr') \
            .text.rsplit('\n', 1)[1]

        # get draft % for all players in top 5 lineups

        # get draft % for last winning lineup

        raise NotImplementedError()
        return {
            'min_winning_score': float(min_winning_score),
            'winning_score': float(winning_score),
        }


    @staticmethod
    def get_contest_identifiers(entry_info) -> tuple[str, tuple, str, str]:
        return (
            f"fd-{entry_info.Sport}-{entry_info.Date:%Y%m%d}-{entry_info.Title}",
            (entry_info.Sport, entry_info.Date, entry_info.Title),
            entry_info.Link,
            "Scoring for " + entry_info.Title,
        )