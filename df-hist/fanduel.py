from collections import defaultdict
import glob
import logging
import os
import time

from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Fanduel(ServiceDataRetriever):
    SERVICE_URL = "https://www.fanduel.com"
    POST_LOGIN_URLS: list[str] = [
        "https://www.fanduel.com/history",
    ]
    LOC_SIGN_IN = (By.LINK_TEXT, "Log in")
    LOC_LOGGED_IN = (By.LINK_TEXT, "Lobby")

    # use longer waits for captcha
    LOGIN_TIMEOUT = 300
    WAIT_TIMEOUT = 300

    _COLUMN_RENAMES = {
        'Date': 'date',
        'Sport': 'sport',
        'Link': 'link',
        'Score': 'score',
        'Entry Id': 'entry_id',
        'Winnings ($)': 'winnings',
        'Position': 'rank',
        'Title': 'title',
        'Entries': 'entries',
        'Entry ($)': 'fee',
    }

    def get_historic_entries_df_from_file(self, history_file_dir):
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
        entries_df = entries_df[entries_df.Date.notna() & entries_df.Entries.notna()]
        if (invalid_dates := rows_of_data - len(entries_df)) > 0:
            LOGGER.info("%i invalid dates found. dropped those entries", invalid_dates)
            rows_of_data = len(entries_df)
        entries_df['contest_id'] = entries_df.Link
        entries_df = entries_df.rename(columns=self._COLUMN_RENAMES)
        entries_df.entries = entries_df.entries.astype(int)
        return entries_df

    def get_entry_lineup_data(self, link, title):
        """ return the HTML for the entry lineup """
        self.browse_to(link, title=title)

        # get draft % for all players in my lineup
        LOGGER.info("waiting for my lineup")
        my_lineup_element = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//div[@data-test-id="contest-entry"]')),
            "Waiting for lineup"
        )

        return my_lineup_element.get_attribute('innerHTML')

    def _get_opponent_lineup_data(self, placement, row_ele):
        """
        similar to get_entry_lineup_data but for other contestents

        placement - a string like '1st', '2nd', '3rd' etc...
        """
        LOGGER.info("scrolling/finding %s place lineup row into view", placement)
        self.pause(f"getting {placement} place lineup")
        self.browser.execute_script('arguments[0].scrollIntoView({block: "center"})', row_ele)
        time.sleep(.3)
        row_ele.click()
        LOGGER.info("Waiting for %s place lineup retrieval", placement)

        opp_lineup_ele = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH, f'//div[@data-test-id="contest-entry"]//span[text()="{placement}"]/ancestor::div[@data-test-id="contest-entry"]')),
            f"Waiting for {placement} place lineup"
        )

        # scroll to end of lineup (really I'm a human)
        LOGGER.info("Scrolling to bottom of lineup")
        self.browser.execute_script('arguments[0].scrollIntoView({block: "start"})', opp_lineup_ele)
        time.sleep(.3)
        self.browser.execute_script('arguments[0].scrollIntoView({block: "center"})', opp_lineup_ele)
        time.sleep(.3)
        self.browser.execute_script('arguments[0].scrollIntoView({block: "end"})', opp_lineup_ele)

        return opp_lineup_ele.get_attribute('innerHTML')

    def _get_last_winning_lineup_data(self, score) -> tuple[int, str]:
        """
        score - the last winning score for the contest
        returns tuple(placement, lineup html)
        """
        self.pause("getting last winning lineup")
        last_winner_link_ele = self.browser.find_element_by_xpath('//button[text()="Last winning position"]')
        LOGGER.info("scrolling/finding last winning lineup link into view")
        self.browser.execute_script('arguments[0].scrollIntoView({block: "center"})', last_winner_link_ele)
        last_winner_link_ele.click()
        last_winning_row = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH, f'//table[@data-test-id="contest-entry-table"]/tbody/tr/td/span[text()="{score}"]/ancestor::tr')
            ),
            "Waiting for first losing lineup"
        )

        placement = last_winning_row.text.split("\n")[0]
        return int(placement[:-2]), self._get_opponent_lineup_data(placement, last_winning_row)

    def get_entry_lineup_df(self, lineup_data):
        """
        lineup_data - string containing html for a contest entry
        """
        soup = BeautifulSoup(lineup_data, 'html.parser')

        lineup_players = []
        for player_row in soup.contents[1:]:
            position = player_row.find('span', {'data-test-id': "player-position"}).text
            assert len(position) > 0
            name = player_row.find('span', {'data-test-id': "player-display-name"}).text
            assert len(name) > 0
            team_ele = player_row.find('abbr', {'data-test-id': "primary-team"})
            team_name = team_ele['title']
            assert len(team_name) > 0
            team_abbr = team_ele.text.split(' ')[0]
            assert len(team_abbr) > 0
            drafted_pct_text = player_row.find("span", text="DRAFTED").parent.span.text
            assert drafted_pct_text[-1] == '%'
            drafted_pct = float(drafted_pct_text[:-1])
            lineup_players.append({
                'position': position,
                'name': name,
                'team_abbr': team_abbr,
                'team_name': team_name,
                'draft_pct': drafted_pct,
            })

        return pd.DataFrame(lineup_players)

    def get_contest_data(self, link, title, contest_key) -> dict:
        self.browse_to(link, title=title)
        entry_table = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH, '//table[@data-test-id="contest-entry-table"]')
            ),
            "Waiting for first place"
        )

        entry_table_rows = entry_table.find_elements_by_xpath('tbody/tr')

        if entry_table_rows[0].text[0] != '1':
            # make sure that the state of the page is fresh (top of the lineups)
            self.pause("resetting to top winning lineup")
            first_place_ele = self.browser.find_element_by_xpath('//button/span[text()="First"]/..')
            LOGGER.info("scrolling/finding top lineups link")
            self.browser.execute_script('arguments[0].scrollIntoView({block: "center"})', first_place_ele)
            first_place_ele.click()
            WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//table[@data-test-id="contest-entry-table"]/tbody/tr/td/div[text()="1st"]')
                ),
                "Waiting for first place"
            )

            entry_table_rows = self.browser.find_elements_by_xpath('//table[@data-test-id="contest-entry-table"]//tbody/tr')

        min_winning_score_str = self.browser.find_element_by_xpath('//span[@data-test-id="RunningManScore"]').text
        winning_score = float(entry_table_rows[0].text.rsplit('\n', 1)[1])

        lineups_data: list[str] = []
        # add draft % for all players in top 5 lineups
        for row_ele in entry_table_rows[:5]:
            placement = row_ele.find_element_by_tag_name('td').text
            lineup_data = self.get_data(
                contest_key + "-lineup-" + placement,
                self._get_opponent_lineup_data,
                data_type='html',
                func_args=(placement, row_ele)
            )
            lineups_data.append(lineup_data)

        lineup_data = self.get_data(
            contest_key + "-lineup-lastwinner",
            self._get_last_winning_lineup_data,
            data_type="json",
            func_args=(min_winning_score_str, ),
        )
        lineups_data.append(lineup_data[1])
        return {
            'last_winning_score': float(min_winning_score_str),
            'top_score': winning_score,
            'lineups_data': lineups_data,
            'winners': lineup_data[0],
        }

    @staticmethod
    def get_contest_identifiers(entry_info) -> tuple[str, tuple, str, str]:
        return (
            f"fd-{entry_info.sport}-{entry_info.date:%Y%m%d}-{entry_info.title}",
            (entry_info.sport, entry_info.date, entry_info.title),
            entry_info.link,
            "Scoring for " + entry_info.title,
        )