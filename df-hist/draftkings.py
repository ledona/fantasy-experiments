import glob
import logging
import os
import time

from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains

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
        'Places_Paid': 'winners',
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

    def _get_lineup_data(self, contestant_name=None) -> tuple:
        """ 
        contestant_name - If not none then wait for the header to display the contestant name
        return tuple of (header element of rendered linup, element containing rendered lineup)
        """
        if contestant_name is not None:
            raise NotImplementedError()
        return self.browser.find_elements_by_xpath(
            '//label[@id="multiplayer-live-scoring-Rank"]/../../../../div'
        )[:2]

    def _get_opponent_lineup_data(self, row_div) -> str:
        """
        return the HTML for the lineup of the opponent on the row
        """
        opp_rank, opp_name = row_div.text.split("\n", 2)[:2]
        self.pause(f"Pausing before getting lineup for opponent '{opp_name}' ranked #{opp_rank}")
        actions = ActionChains(self.browser)
        actions.move_to_element(row_div) \
               .move_by_offset(0, row_div.size['height']) \
               .click() \
               .perform()
        header_ele, lineup_ele = self._get_lineup_data(opp_name)
        return lineup_ele.get_attribute('innerHTML')

    def _get_last_winning_lineup_data(self, last_winner_placement) -> tuple[int, str]:
        raise NotImplementedError()

    def get_contest_data(self, link, contest_key, entry_info) -> dict:
        self.browse_to(
            link,
            title="DraftKings - " + entry_info.title
        )
        WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH, '//label[@id="multiplayer-live-scoring-Rank"]')
            ),
            "Waiting for contest data to fully load"
        )

        standings_list_ele = self.browser.find_element_by_xpath('//div[div/div/span[text()="Rank"]]/div[position()=2]')

        top_entry_table_rows = standings_list_ele.find_elements_by_xpath('div/div')
        if top_entry_table_rows[0].text.split("\n", 1)[0] != "1":
            # scroll to the top
            self.pause(f"scrolling to top of entries")
            self.browser.execute_script(
                "arguments[0].scroll({top: 0, behavior: 'smooth'})", 
                standings_list_ele
            )
            time.sleep(.5)
            top_entry_table_rows = standings_list_ele.find_elements_by_xpath('div/div')

        assert top_entry_table_rows[0].text.split("\n", 1)[0] == "1"
        winning_score = float(top_entry_table_rows[0].text.rsplit('\n', 1)[-1])

        lineups_data: list[str] = []
        # add draft % for all players in top 5 lineups
        for row_ele in top_entry_table_rows[:5]:
            placement_div, _, __ = row_ele.find_elements_by_xpath('div/div')
            placement = int(placement_div.text)
            lineup_data, src, cache_filepath = self.get_data(
                contest_key + f"-lineup-{placement}",
                self._get_opponent_lineup_data,
                data_type='html',
                func_args=(row_ele, )
            )
            LOGGER.info(
                "Entry lineup for '%s' lineup %i retrieved from %s, cached from/to '%s'",
                entry_info.title, placement, src, cache_filepath
            )
            lineups_data.append(lineup_data)

        (min_winning_score, lineup_data), src, cache_filepath = self.get_data(
            contest_key + "-lineup-lastwinner",
            self._get_last_winning_lineup_data,
            data_type="json",
            func_args=(entry_info.winners, ),
        )
        LOGGER.info(
            "Last winning lineup for '%s' retrieved from %s, cached from/to '%s'",
            entry_info.title, src, cache_filepath
        )
        lineups_data.append(lineup_data)

        return {
            'last_winning_score': min_winning_score,
            'top_score': winning_score,
            'lineups_data': lineups_data,
        }

    @staticmethod
    def get_contest_link(entry_info) -> str:
        return f"https://www.draftkings.com/contest/gamecenter/{entry_info.contest_id}?uc={entry_info.entry_id}#/"

    def get_entry_lineup_data(self, link, title):
        self.browse_to(link, title=title)
        return self._get_lineup_data()[1].get_attribute('innerHTML')

