import glob
import logging
import math
import os
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .service_data_retriever import ServiceDataRetriever

_LOGGER = logging.getLogger(__name__)


class Yahoo(ServiceDataRetriever):
    SERVICE_ABBR = "y"
    SERVICE_URL = "https://sports.yahoo.com/dailyfantasy/"
    LOC_SIGN_IN = (By.LINK_TEXT, "Sign in")
    LOC_LOGGED_IN = (By.XPATH, '//a[@data-tst="subnav-contestshistory"][@title="Completed"]')
    LOGIN_TIMEOUT = 45

    _COLUMN_RENAMES = {
        "Sport": "sport",
        "Title": "title",
        "Entry Count": "entries",
        "Entry Fee": "fee",
        "Winnings": "winnings",
        "Entry Id": "entry_id",
        "Id": "contest_id",
        "Points": "score",
        "Rank": "rank",
    }

    @classmethod
    def get_historic_entries_df_from_file(cls, history_file_dir):
        """return an iterator that yields entries"""
        # get the most recent dk contest entry filename
        glob_pattern = os.path.join(history_file_dir, "Yahoo_DF_my_contest_history.*.csv")
        glob_pattern = os.path.expanduser(glob_pattern)
        history_filenames = glob.glob(glob_pattern)

        if len(history_filenames) == 0:
            raise FileNotFoundError(f"No history files found for '{glob_pattern}'")

        retrieval_date_filenames = defaultdict(list)
        for filename in history_filenames:
            retrieval_date_filenames[filename.rsplit(".", 1)[1][:8]].append(filename)
        most_recent_date = sorted(retrieval_date_filenames.keys())[-1]
        _LOGGER.info("Loading history data from '%s'", retrieval_date_filenames[most_recent_date])
        dfs = (
            pd.read_csv(filename, index_col=False)
            for filename in sorted(retrieval_date_filenames[most_recent_date])
        )
        entries_df = pd.concat(dfs)
        dropped_dup_df = entries_df.drop_duplicates()
        if (dup_rows := len(entries_df) - len(dropped_dup_df)) > 0:
            _LOGGER.info("%i duplicate rows dropped", dup_rows)
            entries_df = dropped_dup_df

        entries_df["date"] = pd.to_datetime(entries_df["Start Date"])
        entries_df.Sport = entries_df.Sport.str.lower()
        entries_df = entries_df.rename(columns=cls._COLUMN_RENAMES).query("entries > 1")
        entries_df.fee = entries_df.fee.str.replace("$", "", regex=False).astype(float)
        entries_df.winnings = entries_df.winnings.str.replace("$", "", regex=False).astype(float)
        return entries_df

    def _get_h2h_contest_data(self, link, contest_key, entry_info) -> None | dict:
        """
        return head to head contest data if page content is for head to head, otherwise return None
        """
        self.browse_to(link)
        score_spans = self.browser.find_elements(
            "xpath", '//div[@class="Grid Pos-r"]//span[@class="ydfs-scoring-points"]'
        )
        if len(score_spans) != 2:
            # either there is an error or this contest is for more than 2 contestants
            return None

        winning_score = max(float(score_spans[0].text), float(score_spans[1].text))

        lineup_data, src, cache_filepath = self.get_data(
            contest_key + "-lineup-opp",
            self.get_entry_lineup_data,
            data_type="html",
            func_args=(link, None),
        )
        _LOGGER.info(
            "Opponent entry lineup for '%s' retrieved from %s, cached from/to '%s'",
            entry_info.title,
            src,
            cache_filepath,
        )

        return {
            "last_winning_score": winning_score,
            "top_score": winning_score,
            "lineups_data": [lineup_data],
            "winners": 1,
        }

    # xpath to the entry-rankings / entry vs entry grid
    _CONTEST_GRID_XPATH = '//div[@class="Grid D(f)"]/div[@data-tst="contest-entry"]/..'
    _XPATH_OPPONENT_LINEUP_ROWS = (
        f'({_CONTEST_GRID_XPATH}/div)[3]//tbody/tr[not(@aria-hidden="true")]'
    )
    _PAGE_LINKS_XPATH = f'({_CONTEST_GRID_XPATH}/div)[3]//div[@class="Grid Cf"]//a'

    def _get_opp_lineup_data(self, opponent_lineup_row_ele, rank, reset_when_done=True) -> str:
        """click on the row, get the html, browser back then return the html"""
        self.pause(
            f"wait before clicking on opp lineup row for rank #{rank}", pause_min=1, pause_max=5
        )
        opponent_lineup_row_ele.click()
        WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, '//div[text()="Dim Common Players"]')),
            "Waiting for opponent lineup",
        )
        html = self.get_entry_lineup_data(None, None)

        if reset_when_done:
            self.pause("wait before going back to contestants page", pause_min=1, pause_max=5)
            self.browser.execute_script("window.history.go(-1)")
            WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//div[text()="Select a team below to compare"]')
                ),
                "Waiting to for reload of lineups",
            )
        return html

    def _wait_for_paging(self, page_link, lineups_per_page, page_number, pause_msg):
        """click on a contestant list page and wait for it to load"""
        self.pause(pause_msg, pause_min=1, pause_max=3)
        page_link.click()

        expected_lineup_ranks_showing_text = f"Showing {1 + lineups_per_page * (page_number - 1):,} - {lineups_per_page * page_number:,}"
        WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located(
                (By.XPATH, f"//h5[text()[contains(.,'{expected_lineup_ranks_showing_text}')]]")
            ),
            "Waiting for last winner row",
        )

    def _go_to_last_winner_page(self, last_winner_page, lineups_per_page):
        """navigate to the page of contestant results that has the last paid winner"""
        while True:
            try:
                last_winner_page_a = self.browser.find_element(
                    "xpath", f"{self._PAGE_LINKS_XPATH}[text()={last_winner_page}]"
                )
                # the link is available
                break
            except NoSuchElementException:
                pass

            page_links = self.browser.find_elements("xpath", self._PAGE_LINKS_XPATH)

            # find the next page to go to
            if (
                (last_page_num := int(page_links[-1].text))
                - (second_last_page_num := int(page_links[-2].text))
            ) > 1:
                # there is a page gap when there are a lot of pages, between the current page and the last page
                next_page_num = second_last_page_num
                next_page_link = page_links[-2]
            else:
                next_page_num = last_page_num
                next_page_link = page_links[-1]

            self._wait_for_paging(
                next_page_link,
                lineups_per_page,
                next_page_num,
                f"click on next page ({next_page_num})",
            )

        self._wait_for_paging(
            last_winner_page_a,
            lineups_per_page,
            last_winner_page,
            f"Click on last winner page {last_winner_page}",
        )

    def _find_last_winning_contestant_data(self, last_winner_placement) -> tuple[str, float]:
        """returns - tuple(html for lineup, score)"""
        try:
            page_lineups_ele = self.browser.find_element(
                "xpath", "//h5[text()[contains(.,'Showing 1 - ')]]"
            )
            lineups_per_page = int(page_lineups_ele.text.rsplit(" ", 1)[1])
            last_winner_page = math.ceil(last_winner_placement / lineups_per_page)
        except NoSuchElementException:
            # there are no additional pages
            last_winner_page = 1

        if last_winner_page > 1:
            self._go_to_last_winner_page(last_winner_page, lineups_per_page)

        try:
            last_winner_row = self.browser.find_element(
                "xpath",
                f"{self._XPATH_OPPONENT_LINEUP_ROWS}/td[position()=1][text()='{last_winner_placement}']/..",
            )
        except NoSuchElementException:
            _LOGGER.warning(
                "Failed to find contestant with ranking %s. Likely a tie. Using last contestant on page instead",
                last_winner_placement,
            )
            for row in reversed(
                self.browser.find_elements("xpath", self._XPATH_OPPONENT_LINEUP_ROWS)
            ):
                if int(row.find_element("tagName", "td").text) < last_winner_placement:
                    last_winner_row = row
                    break
            else:
                raise ValueError("Failed to find a rank higher than the last winning placement")

        last_winner_score = float(last_winner_row.find_elements("tagName", "td")[2].text)
        if "highlight" in last_winner_row.get_attribute("class"):
            # I am the last winner, so lets get the lineup for the next contestant
            _LOGGER.warning(
                "I am the last winner, retrieving the lineup draft %%s for the first loser instead"
            )
            try:
                first_loser_row = self.browser.find_element(
                    "xpath",
                    f"{self._XPATH_OPPONENT_LINEUP_ROWS}/td[position()=1][text()='{last_winner_placement + 1}']/..",
                )
                lineup_data = self._get_opp_lineup_data(
                    first_loser_row, last_winner_placement + 1, reset_when_done=False
                )
            except NoSuchElementException:
                _LOGGER.warning(
                    "First loser row not found on this page. Not worth it. No additional lineup data will be retrieved",
                )
                lineup_data = None
        else:
            lineup_data = self._get_opp_lineup_data(
                last_winner_row, last_winner_placement, reset_when_done=False
            )
        return lineup_data, last_winner_score

    def _get_multi_opponent_contest_data(self, link, contest_key, entry_info) -> dict:
        # we should be on the contest page, with no opponent selected
        self.browse_to(link)
        if entry_info.fee > 0:
            paid_places = int(
                self.browser.find_element(
                    "xpath", '//div[@data-tst="contest-header-payout-places"]'
                ).text.replace(",", "")
            )
        else:
            paid_places = 0

        top_contestant_row = self.browser.find_element(
            "xpath", self._XPATH_OPPONENT_LINEUP_ROWS + "[1]"
        )

        if not top_contestant_row.find_element("tagName", "td").text == "1":
            # can happen if we halted a previous retrieval
            self.browser.refresh()
            top_contestant_row = self.browser.find_element(
                "xpath", self._XPATH_OPPONENT_LINEUP_ROWS + "[1]"
            )
            assert top_contestant_row.find_element("tagName", "td").text == "1"

        winning_score = float(top_contestant_row.find_elements("tagName", "td")[2].text)

        lineups_data = []
        for rank in range(1, min(6, entry_info.entries)):
            if rank == entry_info["rank"]:
                continue
            row_ele = self.browser.find_element(
                "xpath", self._XPATH_OPPONENT_LINEUP_ROWS + f"[{rank}]"
            )

            lineup_data, src, cache_filepath = self.get_data(
                f"{contest_key}-lineup-{rank}",
                self._get_opp_lineup_data,
                data_type="html",
                func_args=(row_ele, rank),
            )
            _LOGGER.info(
                "Opponent lineup for '%s' lineup at rank #%i retrieved from %s, cached from/to '%s'",
                entry_info.title,
                rank,
                src,
                cache_filepath,
            )
            lineups_data.append(lineup_data)

        if paid_places > 0:
            (lineup_data, min_winning_score), src, cache_filepath = self.get_data(
                f"{contest_key}-lineup-{rank}",
                self._find_last_winning_contestant_data,
                data_type="json",
                func_args=(paid_places,),
            )
            if lineup_data is not None:
                lineups_data.append(lineup_data)
            _LOGGER.info(
                "Last winning lineup for '%s' retrieved from %s, cached from/to '%s'",
                entry_info.title,
                src,
                cache_filepath,
            )
        else:
            min_winning_score = None

        return {
            "last_winning_score": min_winning_score,
            "top_score": winning_score,
            "lineups_data": lineups_data,
            "winners": paid_places,
        }

    def get_contest_data(self, link, contest_key, entry_info) -> dict:
        data = (
            self._get_h2h_contest_data(link, contest_key, entry_info)
            if entry_info.entries == 2
            else None
        )

        return data or self._get_multi_opponent_contest_data(link, contest_key, entry_info)

    @staticmethod
    def get_entry_link(entry_info) -> str:
        return f"https://sports.yahoo.com/dailyfantasy/contest/{entry_info.contest_id}/{entry_info.entry_id}"

    def get_entry_lineup_data(self, link, title) -> str:
        """
        return - html for the lineups grid (retrieves all data in the grid, including the positions and both contestant lineups)
        """
        if link is not None:
            self.browse_to(link)

        _LOGGER.info("waiting for lineup")
        my_lineup_element = WebDriverWait(self.browser, self.WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.XPATH, self._CONTEST_GRID_XPATH)),
            "Waiting for my lineup",
        )
        return my_lineup_element.get_attribute("innerHTML")

    def _get_lineup_df_helper(self, players_ele, positions_ele):
        lineup_players = []

        for player_row, pos_row in zip(
            players_ele.div.table.tbody.contents, positions_ele.div.table.tbody.contents
        ):
            drafted_pct_ele = player_row.find("span", **{"title": "Percentage rostered"})
            if drafted_pct_ele is None:
                _LOGGER.warning("Draft %% not available for player: %s", player_row.text)
                drafted_pct = None
            else:
                drafted_pct = float(drafted_pct_ele.text.split("%")[0])

            if "No player selected" in player_row.text:
                continue
            player_name_div = player_row.find("div", **{"data-tst": "player-name"})
            if player_name_div is None:
                continue
            name = player_name_div.a.text
            player_record = {
                "position": pos_row.text,
                "name": name,
                "draft_pct": drafted_pct,
            }

            teams_ele = player_row.find("a", **{"data-tst": "player-matchup"}).contents[0].span
            if "Fw-b" in teams_ele.contents[0]["class"]:
                team_ele = teams_ele.contents[0]
            elif "Fw-b" in teams_ele.contents[-1]["class"]:
                team_ele = teams_ele.contents[-1]
            else:
                _LOGGER.debug("Unable to determine team for player '%s'", name)
                team_ele = None

            if team_ele is not None:
                player_record["team_name"] = team_ele.abbr["title"]
                player_record["team_abbr"] = team_ele.text.split(" ")[0]

            lineup_players.append(player_record)

        return pd.DataFrame(lineup_players)

    def get_opp_lineup_df(self, lineup_data):
        """default to using get_entry_lineup_df"""
        soup = BeautifulSoup(lineup_data, "html.parser")
        assert len(soup.contents) == 3
        return self._get_lineup_df_helper(soup.contents[2], soup.contents[1])

    def get_entry_lineup_df(self, lineup_data):
        soup = BeautifulSoup(lineup_data, "html.parser")
        assert len(soup.contents) == 3
        return self._get_lineup_df_helper(soup.contents[0], soup.contents[1])
