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


class Draftkings(ServiceDataRetriever):
    SERVICE_ABBR = 'dk'
    SERVICE_URL = "https://www.draftkings.com"
    POST_LOGIN_URLS: list[str] = [
        "https://www.draftkings.com/mycontests",
    ]

    LOC_SIGN_IN = (By.NAME, "password")
    LOC_LOGGED_IN = (By.LINK_TEXT, "Lobby")

    _COLUMN_RENAMES = {
        'Sport': 'sport',
        'Points': 'score',
        'Entry_Key': 'entry_id',
        'Winnings ($)': 'winnings',
        'Position': 'rank',
        'Entry': 'title',
        'Contest_Entries': 'entries',
        'Entry_Fee': 'fee',
        'Contest_Key': 'contest_id',
    }

    WAIT_TIMEOUT = 300

    @classmethod
    def get_historic_entries_df_from_file(cls, history_file_dir):
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
        entries_df["date"] = pd.to_datetime(entries_df.Contest_Date_EST)
        entries_df = entries_df.rename(columns=cls._COLUMN_RENAMES)
        return entries_df

    def get_entry_lineup_df(self, lineup_data):
        soup = BeautifulSoup(lineup_data, 'html.parser')

        lineup_players = []
        for player_row in soup.table.tbody.contents:
            position = player_row.contents[0].text
            assert '$' in player_row.contents[1].text, \
                "Expected cost to be in the same cell as name, seperated by '$'"
            name = player_row.contents[1].text.split('$', 1)[0]
            drafted_pct_text = player_row.contents[2].text
            assert drafted_pct_text[-1] == '%'
            team_cell_spans = player_row.contents[3].find_all('span')

            assert (len(team_cell_spans[0]['class']) > 0) != (len(team_cell_spans[3]['class']) > 0), \
                "Expected 1 team to have a class, the player's team"
            team = (
                team_cell_spans[0].text
                if len(team_cell_spans[0]['class']) > 0 else
                team_cell_spans[3].text
            )
            lineup_players.append({
                'position': position,
                'name': name,
                'team_abbr': team,
                'draft_pct': float(drafted_pct_text[:-1]),
            })
        return pd.DataFrame(lineup_players)

    def get_contest_data(self, link, title, contest_key) -> dict:
        self.browse_to(
            link,
            title="DraftKings - " + title
        )
        entry_table = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH, '//label[@id="multiplayer-live-scoring-Rank"]')
            ),
            "Waiting for first place"
        )

        raise NotImplementedError("parse my lineup for draft % and cache it")
        raise NotImplementedError("Parse contest info")
        return {
            'last_winning_score': float(min_winning_score_str),
            'top_score': winning_score,
            'lineups_data': lineups_data,
            'winners': lineup_data[0],
        }

    @staticmethod
    def get_contest_link(entry_info) -> str:
        return f"https://www.draftkings.com/contest/gamecenter/{entry_info.contest_id}?uc={entry_info.entry_id}#/"

    def get_entry_lineup_data(self, link, title):
        self.browse_to(link, title=title)

        # get draft % for all players in my lineup
        my_lineup_element = self.browser.find_elements_by_xpath(
            '//label[@id="multiplayer-live-scoring-Rank"]/../../../../div'
        )[1]
        return my_lineup_element.get_attribute('innerHTML')
