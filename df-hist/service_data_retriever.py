from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod
from importlib import import_module
import functools
import json
import time
import random
from typing import Optional
import logging
import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options as ChromeOptions

LOGGER = logging.getLogger(__name__)

PAUSE_MIN = 10
PAUSE_MAX = 15


class ServiceDataRetriever(ABC):
    # URL to the service home page
    SERVICE_URL: str
    # links to go to after logging in (look like a human)
    POST_LOGIN_URLS: list[str] = []
    # locator for the signin/login link. if present then the account is not logged in
    LOC_SIGN_IN: tuple[str, str]
    LOC_LOGGED_IN: tuple[str, str]

    # how long to wait for login before timing out
    LOGIN_TIMEOUT = 30

    def __init__(
            self, browser_address: Optional[str] = None, browser_debug_port: Optional[bool] = None,
            browser_profile_path: Optional[str] = None, cache_path: Optional[str] = None,
    ):
        """
        browser_address - the connection address of the chrome instance to use. e.g. ip-address:port
        """
        self.cache_path = cache_path

        self.browser_address = browser_address
        self.browser_debug_port = browser_debug_port
        self.browser_profile_path = browser_profile_path

        self._player_draft_dicts: list[dict] = []
        self._contest_dicts: list[dict] = []
        self._entry_dicts: list[dict] = []
        # used to keep track of the contests that have been processed
        self.processed_contests = set()

    @functools.cached_property
    def browser(self):
        chrome_options = ChromeOptions()
        if self.browser_address is not None:
            # connect to existing browser
            assert self.browser_debug_port is None
            assert self.browser_profile_path is None

            LOGGER.info("try and connect to an existing browser at %s", self.browser_address)
            chrome_options.add_experimental_option(
                "debuggerAddress", self.browser_address
            )
        else:
            # try and create a new browser to run the retrieval from
            LOGGER.info("Opening new browser")
            chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("start-maximized")
            if self.browser_debug_port is not None:
                LOGGER.info("Exposing chrome debugging on port '%s'", self.browser_debug_port)
                chrome_options.add_argument(f"--remote-debugging-port={self.browser_debug_port}")
            if self.browser_profile_path is not None:
                LOGGER.info("Using chrome profile at '%s'", self.browser_profile_path)
                chrome_options.add_argument(f"--user-data-dir={self.browser_profile_path}")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-software-rasterizer")
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')

        browser_ = webdriver.Chrome("chromedriver", options=chrome_options)
        try:
            browser_.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as ex:
            LOGGER.error("Error updating browser webdriver property", exc_info=ex)
        return browser_

    @property
    def player_draft_df(self):
        """
        returns - dataframe with columns: contest_id, date, sport, team, position, first name, last name,
            draft position, draft %, score
        """
        df = pd.DataFrame(self._player_draft_dicts)
        assert set(df.columns) == {
            'contest_id', 'date', 'sport', 'team', 'position', 'draft_position',
            'score', 'first_name', 'last_name', 'draft_%'
        }
        return df

    @property
    def contest_df(self):
        """
        returns - dataframe with columns: contest_id, date, sport, contest_name,
           entry_fee, entry_count, winning_places, top_score, last_winning_score, last_winner_rank
        """
        df = pd.DataFrame(self._contest_dicts)
        assert set(df.columns) == {
            'contest_id', 'date', 'sport', 'contest_name', 'entry_fee', 'entry_count',
            'winning_places', 'top_score', 'last_winning_score', 'last_winning_rank',
        }
        return df

    @property
    def entry_df(self):
        """
        returns - dataframe with columns: contest_id, entry_id, link, winnings, rank, score
        """
        df = pd.DataFrame(self._entry_dicts)
        assert set(df.columns) == {'contest_id', 'entry_id', 'link', 'winnings', 'rank', 'score'}
        return df

    def wait_on_login(self):
        LOGGER.info(
            "Running login flow... going to '%s' if not already there",
            self.SERVICE_URL
        )
        if self.browser.current_url != self.SERVICE_URL:
            self.browser.get(self.SERVICE_URL)

        try:
            # if found, then signin is required
            self.browser.find_element(*self.LOC_SIGN_IN)
        except NoSuchElementException:
            LOGGER.info("Log in link %s not found. Assuming account is already logged in",
                        self.LOC_SIGN_IN)
            return

        LOGGER.info("Waiting %i seconds for you to log in...", self.LOGIN_TIMEOUT)
        WebDriverWait(self.browser, self.LOGIN_TIMEOUT).until(
            EC.presence_of_element_located(self.LOC_LOGGED_IN),
            "Still not logged in..."
        )

    @property
    def logged_in_to_service(self) -> bool:
        """ use self.LOC_LOGGED_IN to confirm that the account is logged in """
        try:
            # if found, then signin succeeded
            WebDriverWait(self.browser, self.LOGIN_TIMEOUT).until(
                EC.presence_of_element_located(self.LOC_LOGGED_IN),
                "Still not logged in..."
            )
            return True
        except NoSuchElementException:
            LOGGER.error("Logged in link %s not found! Confirmation of logged in status failed",
                         self.LOC_LOGGED_IN)
            return False

    @abstractclassmethod
    def get_entries_df(cls, history_file_dir):
        """
        return a dataframe with entries data, the dataframe must include the following columns
           Sport - lower case sport abbreviation
           Date - column with date.date objects
        """
        raise NotImplementedError()

    @abstractmethod
    def get_entry_lineup_data(self, link, title) -> str:
        """ retrieve entry data from link. returns a dict containing unprocessed/raw data that can be cached and processed """
        raise NotImplementedError()

    @abstractmethod
    def get_entry_lineup_df(self, lineup_data) -> pd.DataFrame:
        """ process the entry data returned by get_entry_data, return a dataframe """
        raise NotImplementedError()

    def add_entry_data(self, lineup_df, entry_info):
        """ add entry lineup and info to player draft and entry datasets """
        raise NotImplementedError()

    @abstractmethod
    def get_contest_data(self, link, title) -> dict:
        """ 
        get all contest data that is not in the entry info, return a dict containing that data 
        dict is expected to have a key named 'lineups' that contains lineup data for player draft
        """
        raise NotImplementedError()

    def add_entry_info(self, entry_info):
        raise NotImplementedError("add entry information to the entry dataset")

    def process_entry(self, entry_info):
        """
        Process a contest entry.
        If the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        self.add_entry_info(entry_info)

        contest_key, contest_id, link, title = self.get_contest_identifiers(entry_info)
        entry_key = contest_key + '-' + entry_info['Entry Id']

        # handle entry lineup info
        entry_lineup_data = self.get_data(entry_key, self.get_entry_lineup_data, link, title, data_type='txt')
        entry_lineup_df = self.get_entry_lineup_df(entry_lineup_data)
        self.add_lineup_to_draft_df(entry_lineup_df)

        # if contest has been processed then we are done
        if contest_id in self.processed_contests:
            return

        # process contest data
        contest_data = self.get_data(contest_key, self.get_contest_data, link, title, data_type='json')
        for lineup_data in contest_data['lineups']:
            lineup_df = self.get_entry_lineup_df(lineup_data)
            self.add_lineup_to_draft_df(lineup_df)

        self.processed_contests.add(contest_id)

    @abstractstaticmethod
    def get_contest_identifiers(entry_info) -> tuple[str, tuple, str, str]:
        """ returns - (contest key, contest id, entry link, browser page title) """
        raise NotImplementedError()

    def browse_to(self, url, pause_before=True, title=None):
        """
        browse to the requested page, possibly with a pause before browsing
        if the current browser title is the same as title OR the current browser URL is the same as url then
        don't load anything new, use the current page content
        """
        LOGGER.info("Browsing to url='%s' title='%s'", url, title)
        # first check to see if we are already on that page
        if self.browser.current_url == url or self.browser.title == title:
            LOGGER.info("Browser is already at url='%s', title='%s'", url, title)
            return

        if not self.logged_in_to_service:
            self.wait_on_login()
            for i, url in enumerate(self.POST_LOGIN_URLS, 1):
                if pause_before:
                    pause_for = random.randint(PAUSE_MIN, PAUSE_MAX)
                    LOGGER.info("Pausing for %i seconds before post login link #%igetting url content", pause_for, i)
                    time.sleep(pause_for)
                LOGGER.info("Going to post login url #%i '%s'", i, url)
                self.browser.get(url)

        if pause_before:
            pause_for = random.randint(PAUSE_MIN, PAUSE_MAX)
            LOGGER.info("Pausing for %i seconds before getting url content", pause_for)
            time.sleep(pause_for)

        LOGGER.info("Getting content at '%s'", url)
        self.browser.get(url)

    def get_data(self, cache_key: str, func: callable, link, title=None, data_type: str = 'csv'):
        """
        get data related to the key, first try the cache, if that fails call func
        and cache the result before returning result

        title - the title of the expected browser page, used to skip getting new browser content
        type - the type of data to be loaded. csv -> dataframe, json -> dict/list, txt -> str.
            This should be the same as the data type returned by func
        func - a function that when executed returns the required data, takes as arguments, (link, title)
        """
        # see if it in the cache
        if self.cache_path is not None and \
           os.path.isfile(cache_filepath := os.path.join(self.cache_path, f"{cache_key}.{data_type}")):
            if data_type == 'csv':
                return pd.read_csv(cache_filepath)
            elif data_type == 'json':
                with open(cache_filepath, "r") as f_:
                    return json.load(f_)
            elif data_type == 'txt':
                with open(cache_filepath, "r") as f_:
                    return f_.read()

            raise ValueError(f"Don't know how to load '{cache_filepath}' from cache")

        data = func(link, title)

        if self.cache_path is not None:
            if data_type == 'csv':
                data.to_csv(cache_filepath)
            elif data_type == 'json':
                with open(cache_filepath, "w") as f_:
                    json.save(f_, data)
            elif data_type == 'txt':
                with open(cache_filepath, "w") as f_:
                    f_.write(data)
            else:
                raise ValueError(f"Don't know how to write '{cache_filepath}' to cache")

        return data

def get_service_data_retriever(
        service: str,
        cache_path: Optional[str] = None,
        browser_address: Optional[str] = None,
        browser_debug_port: Optional[str] = None,
        browser_profile_path: Optional[str] = None,
) -> ServiceDataRetriever:
    """ attempt to import the data retriever for the requested service """
    module = import_module(service)
    class_ = getattr(module, service.capitalize())
    if cache_path is not None and not os.path.isdir(cache_path):
        raise FileNotFoundError(f"cache path '{cache_path}' does not exist!")
    return class_(
        browser_address=browser_address,
        browser_debug_port=browser_debug_port,
        browser_profile_path=browser_profile_path,
        cache_path=cache_path,
    )
