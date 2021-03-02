import glob
import logging
import os

from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Yahoo(ServiceDataRetriever):
    SERVICE_ABBR = 'y'
    SERVICE_URL = "https://sports.yahoo.com/dailyfantasy/"
    LOC_SIGN_IN = (By.LINK_TEXT, "Sign in")
    LOC_LOGGED_IN = (By.LINK_TEXT, "Create contest")
    LOGIN_TIMEOUT = 45

    _COLUMN_RENAMES = {
        'Sport': 'sport',
        'Title': 'title',
        'Entry Count': 'entries',
        'Entry Fee': 'fee',
        'Winnings': 'winnings',
        'Entry Id': 'entry_id',
        'Id': 'contest_id',
        'Points': 'score',
        'Rank': 'rank',
    }

    @classmethod
    def get_historic_entries_df_from_file(cls, history_file_dir):
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
        entries_df["date"] = pd.to_datetime(entries_df['Start Date'])
        entries_df.Sport = entries_df.Sport.str.lower()
        entries_df = entries_df.rename(columns=cls._COLUMN_RENAMES)
        return entries_df

    def get_contest_data(self, link, contest_key, entry_info) -> dict:
        raise NotImplementedError()

    @staticmethod
    def get_contest_link(entry_info) -> str:
        return f"https://sports.yahoo.com/dailyfantasy/contest/{entry_info.contest_id}/{entry_info.entry_id}"

    def get_entry_lineup_data(self, link, title):
        self.browse_to(link, title=title)

        LOGGER.info("waiting for my lineup")
        my_lineup_element = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//div[@class="Grid D(f)"]/div[@data-tst="contest-entry"]/..')),
            "Waiting for my lineup"
        )

        return my_lineup_element.get_attribute('innerHTML')

    def get_entry_lineup_df(self, lineup_data):
        soup = BeautifulSoup(lineup_data, 'html.parser')

        lineup_players = []
    
        for player_row, pos_ele in zip(soup.contents[0].div.table.tbody.contents, 
                                       soup.contents[1].div.table.tbody.contents):
            raise NotImplementedError()

        raise NotImplementedError()
