import glob
import logging
import os
import time
from typing import Optional

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
        'Place': 'rank',
        'Entry': 'title',
        'Contest_Entries': 'entries',
        'Entry_Fee': 'fee',
        'Contest_Key': 'contest_id',
        'Places_Paid': 'winners',
    }

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
        entries_df['winnings'] = (
            entries_df.Winnings_Non_Ticket.str.replace('$', '', regex=False).astype(float) +
            entries_df.Winnings_Ticket.str.replace('$', '', regex=False).astype(float)
        )
        return entries_df

    @staticmethod
    def _draft_percentages(player_row, lineup_position) -> Optional[list[tuple[float, Optional[str]]]]:
        """
        parse the soup coutents of a player row, return percentages and associated lineup position
        lineup_position - the position of the player in the lineup, use this for single position percentage data
            ignored for multiple pct data
        return - list of tuples of (draft percentage, lineup position)
        """
        drafted_pct_text = player_row.contents[2].text
        if drafted_pct_text == '--':
            return []
        if drafted_pct_text.count('%') == 1:
            assert drafted_pct_text[-1] == '%'
            return [(float(drafted_pct_text[:-1]), lineup_position)]

        drafted_pcts = []
        # there are multiple draft percentages
        for drafted_pct_ele in player_row.contents[2].div.contents:
            assert len(drafted_pct_ele.contents) in (1, 2), \
                "Expecting either 2 elements (position, pct) or just an element with pct!"
            if len(drafted_pct_ele.contents) == 1:
                drafted_pcts.append((float(drafted_pct_ele.text[:-1]), None))
            else:
                drafted_pcts.append((
                    float(drafted_pct_ele.contents[1].text[:-1]),
                    drafted_pct_ele.contents[0].text
                ))
        return drafted_pcts

    def get_entry_lineup_df(self, lineup_data):
        soup = BeautifulSoup(lineup_data, 'html.parser')
        if soup.table.tbody is None:
            if 'Failed to draft in time' not in soup.text:
                raise ValueError("Failed to parse entry lineup. Missing tbody")
            return None

        lineup_players = []
        for player_row in soup.table.tbody.contents:
            position = player_row.contents[0].text
            name = player_row.contents[1].text
            if '$' in name:
                name = name.split('$', 1)[0]
            else:
                LOGGER.warning("Player cost did not appear with name, maybe processed before cost was retrieved...")
            team_cell_spans = player_row.contents[3].find_all('span')

            if len(team_cell_spans) in [7, 8]:
                # this happens when there is a starters indication, drop the first 2 spans to get AWAY @ HOME
                team_cell_spans = team_cell_spans[2:]
                assert team_cell_spans[2].text.strip() == '@'

            assert len(team_cell_spans[0].text) > 0, "First item should be a team abbr"

            # there are either 3 or 5 spans. 5 if the player's game happened, 3 if it was postponed (no spans for score)
            if len(team_cell_spans) in (5, 6):
                assert team_cell_spans[2].text.strip() in ['@', 'v']
                team_2_span_idx = 3
            else:
                assert len(team_cell_spans) == 3, "Expected there to be 3 spans!"
                assert team_cell_spans[1].text.strip() in ['@', 'v']
                team_2_span_idx = 2

            # for the following assert the index of team 2's abbr span is based on the number of spans
            assert (
                (len(team_cell_spans[0]['class']) > 0) !=
                (len(team_cell_spans[team_2_span_idx]['class']) > 0)
            ), "Expected 1 team to have a class, the player's team"
            team = (
                team_cell_spans[0].text
                if len(team_cell_spans[0]['class']) > 0 else
                team_cell_spans[team_2_span_idx].text
            )

            for (draft_pct, draft_position) in self._draft_percentages(player_row, position):
                lineup_players.append({
                    'position': draft_position,
                    'name': name,
                    'team_abbr': team,
                    'draft_pct': draft_pct,
                })

        return pd.DataFrame(lineup_players)

    def _get_lineup_data(self, contestant_name=None) -> tuple:
        """
        contestant_name - If not none then wait for the header to display the contestant name
        return tuple of (header element of rendered linup, element containing rendered lineup)
        """
        if contestant_name is not None:
            WebDriverWait(self.browser, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, f'//div[@role="heading"]/div/div[text()="{contestant_name}"]')
                ),
                f"Waiting for opposing content's lineup to load. {contestant_name=}"
            )

        lineup_data = WebDriverWait(self.browser, 10).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, '//label[@id="multiplayer-live-scoring-Rank"]/../../../../div')
            ),
            "Waiting for lineup data to load"
        )
        assert len(lineup_data) >= 2
        return lineup_data[:2]

    def _get_opponent_lineup_data(self, row_div) -> str:
        """
        return the HTML for the lineup of the opponent on the row
        """
        opp_rank, opp_name = row_div.text.split("\n", 2)[:2]
        self.pause(f"wait before getting lineup for opponent '{opp_name}' ranked #{opp_rank}")
        row_div.click()
        lineup_ele = self._get_lineup_data(opp_name)[1]
        return lineup_ele.get_attribute('innerHTML')

    def _get_last_winning_lineup_data(self, last_winner_rank, standings_list_ele) -> tuple[int, str]:
        self.pause(f"scroll to last winner ranked {last_winner_rank}")
        list_height = standings_list_ele.size['height']
        position = self.browser.execute_script("return arguments[0].scrollTop", standings_list_ele)

        # iterate while the rank of the last row is less than the desired rank
        prev_last_row_rank = -1
        while True:
            rows = standings_list_ele.find_elements_by_xpath('div/div')
            last_row_rank = int(rows[-1].text.split("\n", 1)[0])
            if last_row_rank > last_winner_rank or prev_last_row_rank == last_row_rank:
                break
            prev_last_row_rank = last_row_rank
            position += list_height
            self.browser.execute_script(
                "arguments[0].scroll({top: arguments[1]})",
                standings_list_ele, position
            )

        # iterate through rows in reverse order till we find the first rank <= last_winner_rank
        for row_ele in reversed(rows):
            if int(row_ele.text.split("\n", 1)[0]) <= last_winner_rank:
                score = float(row_ele.text.rsplit("\n", 1)[-1].replace(',', ''))
                return score, self._get_opponent_lineup_data(row_ele)

        raise ValueError("Unable to find last winner")

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

        standings_list_ele = self.browser.find_element_by_xpath(
            '//div[div/div/span[text()="Rank"]]/div[position()=2]'
        )

        top_entry_table_rows = standings_list_ele.find_elements_by_xpath('div/div')
        if top_entry_table_rows[0].text.split("\n", 1)[0] != "1":
            # scroll to the top
            self.pause("scroll to top of entries")
            self.browser.execute_script(
                "arguments[0].scroll({top: 0})",
                standings_list_ele
            )
            top_entry_table_rows = standings_list_ele.find_elements_by_xpath('div/div')

        assert top_entry_table_rows[0].text.split("\n", 1)[0] == "1"
        winning_score = float(top_entry_table_rows[0].text.rsplit('\n', 1)[-1].replace(',', ''))

        lineups_data: list[str] = []
        # add draft % for all players in top 5 lineups
        for row_ele in top_entry_table_rows[:5]:
            placement_div = row_ele.find_element_by_xpath('div/div[1]')
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

        if entry_info.winners > 0:
            (min_winning_score, lineup_data), src, cache_filepath = self.get_data(
                contest_key + "-lineup-lastwinner",
                self._get_last_winning_lineup_data,
                data_type="json",
                func_args=(entry_info.winners, standings_list_ele),
            )
            LOGGER.info(
                "Last winning lineup for '%s' retrieved from %s, cached from/to '%s'",
                entry_info.title, src, cache_filepath
            )
            lineups_data.append(lineup_data)
        else:
            min_winning_score = 0

        return {
            'last_winning_score': min_winning_score,
            'top_score': winning_score,
            'lineups_data': lineups_data,
        }

    @staticmethod
    def get_entry_link(entry_info) -> str:
        return f"https://www.draftkings.com/contest/gamecenter/{entry_info.contest_id}?uc={entry_info.entry_id}#/"

    def get_entry_lineup_data(self, link, title):
        self.browse_to(link, title=title)
        lineup_data = self._get_lineup_data()
        return lineup_data[1].get_attribute('innerHTML')

    def is_entry_supported(self, entry_info) -> Optional[str]:
        if entry_info.entries == 2:
            return "H2H contests with only 2 entries is not supported"
        return None
