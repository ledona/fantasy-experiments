import glob
import logging
import math
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
    LOC_LOGGED_IN = (By.XPATH, '//a[@data-tst="subnav-contestshistory"][@title="Completed"]')
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

    def _get_h2h_contest_data(self, link, contest_key, entry_info) -> dict:
        self.browse_to(link)
        score_spans = self.browser.find_elements_by_xpath(
            '//div[@class="Grid Pos-r"]//span[@class="ydfs-scoring-points"]'
        )
        assert len(score_spans) == 2
        winning_score = max(float(score_spans[0].text), float(score_spans[1].text))

        lineup_data, src, cache_filepath = self.get_data(
            contest_key + "-lineup-opp",
            self.get_entry_lineup_data,
            data_type='html',
            func_args=(link, None)
        )
        LOGGER.info(
            "Opponent entry lineup for '%s' retrieved from %s, cached from/to '%s'",
            entry_info.title, src, cache_filepath
        )

        return {
            'last_winning_score': winning_score,
            'top_score': winning_score,
            'lineups_data': [lineup_data],
            'winners': 1,
        }

    # xpath to the entry-rankings / entry vs entry grid
    _CONTEST_GRID_XPATH = '//div[@class="Grid D(f)"]/div[@data-tst="contest-entry"]/..'
    _XPATH_OPPONENT_LINEUP_ROWS = f'({_CONTEST_GRID_XPATH}/div)[3]//tbody/tr[not(@aria-hidden="true")]'
    def _get_opp_lineup_data(self, opponent_lineup_row_ele) -> str:
        """ click on the row, get the html, browser back then return the html """
        opponent_lineup_row_ele.click()
        self.pause("Waiting after request for opponent lineup")
        WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//div[text()="Dim Common Players"]')),
            "Waiting for opponent lineup"
        )
        html = self.get_entry_lineup_data(None, None)
        self.browser.execute_script("window.history.go(-1)")
        return html

    def _get_opponent_rows(self, my_name):
        rows = self.browser.find_elements_by_xpath(self._XPATH_OPPONENT_LINEUP_ROWS)
        raise NotImplementedError("drop the last row if it is for me")
        return rows

    def _find_min_winning_contestant_data(self, last_winner_placement) -> tuple[str, float]:
        # figure out my name
        raise NotImplementedError()

        rows = self._get_opponent_rows(my_name)
        if int(rows[-1].find_element_by_tag_name('td').text) < last_winner_placement:
            # figure out which page the last winner is on
            last_winner_page = math.ceil(last_winner_placement / len(rows))
            # figure out the last page number
            raise NotImplementedError()
            if (last_page_number / 2) > last_winner_page:
                # page up to the last winner page from the beginning
                raise NotImplementedError()
            else:
                # page down to the last winner page from the end
                raise NotImplementedError()
            rows = self._get_opponent_rows(my_name)

        # test to make sure the last winner is on this page, warn if something looks hinky
        raise NotImplementedError()

        raise NotImplementedError("find min winning score and the associated lineup")

    def _get_multi_opponent_contest_data(self, link, contest_key, entry_info) -> dict:
        # we should be on the contest page, with no opponent selected
        self.browse_to(link)
        paid_places = int(self.browser.find_element_by_xpath(
            '//div[@data-tst="contest-header-payout-places"]'
        ).text)

        top_contestant_row = self.browser.find_element_by_xpath(self._XPATH_OPPONENT_LINEUP_ROWS + "[1]")
        assert top_contestant_row.find_element_by_tag_name('td').text == "1"
        winning_score = float(top_contestant_row.find_elements_by_tag_name('td')[2].text)

        lineups_data = []
        for rank in range(1, 6):
            row_ele = self.browser.find_element_by_xpath(
                self._XPATH_OPPONENT_LINEUP_ROWS + f'[{rank}]'
            )
            assert int(row_ele.find_element_by_tag_name("td").text) == rank
            lineup_data, src, cache_filepath = self.get_data(
                f"{contest_key}-lineup-{rank}",
                self._get_opp_lineup_data,
                data_type='html',
                func_args=(row_ele, )
            )
            LOGGER.info(
                "Opponent lineup for '%s' lineup at rank #%i retrieved from %s, cached from/to '%s'",
                entry_info.title, rank, src, cache_filepath
            )
            lineups_data.append(lineup_data)

        (lineup_data, min_winning_score), src, cache_filepath = self.get_data(
            f"{contest_key}-lineup-{rank}",
            self._find_min_winning_contestant_data,
            data_type='json',
            func_args=(paid_places, )
        )

        return {
            'last_winning_score': min_winning_score,
            'top_score': winning_score,
            'lineups_data': lineups_data,
            'winners': paid_places,
        }

    def get_contest_data(self, link, contest_key, entry_info) -> dict:
        return (
            self._get_h2h_contest_data(link, contest_key, entry_info)
            if entry_info.entries == 2 else
            self._get_multi_opponent_contest_data(link, contest_key, entry_info)
        )

    @staticmethod
    def get_entry_link(entry_info) -> str:
        return f"https://sports.yahoo.com/dailyfantasy/contest/{entry_info.contest_id}/{entry_info.entry_id}"

    def get_entry_lineup_data(self, link, title) -> str:
        """
        return - html for the lineups grid (retrieves all data in the grid, including the positions and both contestant lineups)
        """
        if link is not None:
            self.browse_to(link)

        LOGGER.info("waiting for lineup")
        my_lineup_element = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, self._CONTEST_GRID_XPATH)),
            "Waiting for my lineup"
        )
        return my_lineup_element.get_attribute('innerHTML')

    def _get_lineup_df_helper(self, players_ele, positions_ele):
        lineup_players = []

        for player_row, pos_row in zip(players_ele.div.table.tbody.contents,
                                       positions_ele.div.table.tbody.contents):
            drafted_pct_ele = player_row.find("span", **{'title': 'Percentage rostered'})
            assert drafted_pct_ele.text.endswith("% Rostered")
            drafted_pct = float(drafted_pct_ele.text.split('%')[0])

            name = player_row.find("div", **{'data-tst': "player-name"}).a.text
            player_record = {
                'position': pos_row.text,
                'name': name,
                'draft_pct': drafted_pct,
            }

            teams_ele = player_row.find("a", **{'data-tst': "player-matchup"}) \
                                  .contents[0] \
                                  .span
            if 'Fw-b' in teams_ele.contents[0]['class']:
                team_ele = teams_ele.contents[0]
            elif 'Fw-b' in teams_ele.contents[-1]['class']:
                team_ele = teams_ele.contents[-1]
            else:
                LOGGER.warning("Unable to determine team for player '%s'", name)
                team_ele = None

            if team_ele is not None:
                player_record['team_name'] = team_ele.abbr['title']
                player_record['team_abbr'] = team_ele.text.split(' ')[0]

            lineup_players.append(player_record)

        return pd.DataFrame(lineup_players)

    def get_opp_lineup_df(self, lineup_data):
        """ default to using get_entry_lineup_df """
        soup = BeautifulSoup(lineup_data, 'html.parser')
        assert len(soup.contents) == 3
        return self._get_lineup_df_helper(
            soup.contents[2],
            soup.contents[1]
        )

    def get_entry_lineup_df(self, lineup_data):
        soup = BeautifulSoup(lineup_data, 'html.parser')
        assert len(soup.contents) == 3
        return self._get_lineup_df_helper(
            soup.contents[0],
            soup.contents[1]
        )