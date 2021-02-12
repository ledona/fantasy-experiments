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

PAUSE_MIN = 5
PAUSE_MAX = 60

# columns that will be present in the outputted contest dataframe
EXPECTED_CONTEST_COLS = {
    'contest_id', 'date', 'sport', 'title', 'fee', 'entries',
    'winners', 'top_score', 'last_winning_score',
}

# columns that will be in the outputted entries dataframe
EXPECTED_ENTRIES_COLS = {
    'contest_id', 'entry_id', 'link', 'winnings', 'rank', 'score',
    'date',   # column with date.date objects
    'sport',  # lower case sport abbreviation
}

# columns that must exist in dataframe based on historic entries data files
EXPECTED_HISTORIC_ENTRIES_DF_COLS = {
    'contest_id', 'date', 'sport', 'title', 'fee', 'entries',
}

# columns that will be in the player draft percentage dataframe
EXPECTED_DRAFT_PLAYER_COLS = {
    'contest', 'date', 'sport', 'team_abbr', 'team_name', 'position',
    'name', 'draft_pct'
}


EXPECTED_CONTEST_DATA_KEYS = {
    'lineups_data',  # list of lineup data strings
    'winners', 'top_score', 'last_winning_score',
}


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
            interactive = False,
    ):
        """
        browser_address - the connection address of the chrome instance to use. e.g. ip-address:port
        interactive - if true require user confirmation prior to every browser action
        """
        self.cache_path = cache_path
        self.interactive = interactive

        self.browser_address = browser_address
        self.browser_debug_port = browser_debug_port
        self.browser_profile_path = browser_profile_path

        self._player_draft_dfs: list[pd.DataFrame] = []
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
        returns - dataframe with columns in EXPECTED_DRAFT_PLAYER_COLS
        """
        df = pd.concat(self._player_draft_dfs, ignore_index=True) \
               .drop_duplicates(ignore_index=True)
        assert set(df.columns) == EXPECTED_DRAFT_PLAYER_COLS
        return df

    @property
    def contest_df(self):
        """
        returns - dataframe with columns matching EXPECTED_CONTEST_COLS
        """
        df = pd.DataFrame(self._contest_dicts)
        assert set(df.columns) <= EXPECTED_CONTEST_COLS, \
            f"Missing columns: {EXPECTED_CONTEST_COLS - set(df.columns)}"
        return df

    @property
    def entry_df(self):
        """
        returns - dataframe with columns: contest_id, entry_id, link, winnings, rank, score
        """
        df = pd.DataFrame(self._entry_dicts)
        assert EXPECTED_ENTRIES_COLS <= set(df.columns), \
            f"Missing cols: {EXPECTED_ENTRIES_COLS - set(df.columns)}"
        return df

    def wait_on_login(self):
        LOGGER.info(
            "Running login flow... going to '%s' if not already there",
            self.SERVICE_URL
        )
        if self.browser.current_url != self.SERVICE_URL:
            if self.interactive:
                input(f"About to retrieve {self.SERVICE_URL}. <Enter> to continue")
            self.browser.get(self.SERVICE_URL)

        try:
            # if found, then signin is required
            self.browser.find_element(*self.LOC_SIGN_IN)
        except NoSuchElementException:
            LOGGER.info("Log in link %s not found. Assuming account is already logged in",
                        self.LOC_SIGN_IN)
            return

        if self.interactive:
            input("Waiting for you to log in. <Enter> to continue")

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
    def get_historic_entries_df_from_file(cls, history_file_dir):
        """
        return a dataframe with entries data retrieved from the service's entry history data files
            The dataframe must include all columns named in EXPECTED_HISTORIC_ENTRIES_DF_COLS.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_entry_lineup_data(self, link, title) -> str:
        """ retrieve entry data from link. returns a dict containing unprocessed/raw data that can be cached and processed """
        raise NotImplementedError()

    @abstractmethod
    def get_entry_lineup_df(self, lineup_data) -> pd.DataFrame:
        """
        process the entry data returned by get_entry_data, return a dataframe
        returned dataframe should have columns position, name, team_abbr, team_name, drafted_pct
        """
        raise NotImplementedError()

    @abstractmethod
    def get_contest_data(self, link, title, contest_key) -> dict:
        """
        get all contest data that is not in the entry info and return in a dict
        """
        raise NotImplementedError()

    def process_entry(self, entry_info):
        """
        Process a contest entry.
        If the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        LOGGER.info(
            "Processing %s %s '%s'",
            entry_info.date.strftime("%Y%m%d"), entry_info.sport, entry_info.title
        )
        entry_dict = {
            name: entry_info.get(name)
            for name in EXPECTED_ENTRIES_COLS
        }
        self._entry_dicts.append(entry_dict)

        contest_key, contest_id, link, title = self.get_contest_identifiers(entry_info)
        entry_key = contest_key + '-entry-' + entry_info.entry_id

        # handle entry lineup info
        entry_lineup_data = self.get_data(
            entry_key, self.get_entry_lineup_data, data_type='html',
            func_args=(link, title),
        )
        entry_lineup_df = self.get_entry_lineup_df(entry_lineup_data)
        entry_lineup_df['contest'] = contest_key
        entry_lineup_df['date'] = entry_info.date
        entry_lineup_df['sport'] = entry_info.sport
        self._player_draft_dfs.append(entry_lineup_df)

        # if contest has been processed then we are done
        if contest_id in self.processed_contests:
            return

        # process contest data
        contest_data = self.get_data(
            contest_key, self.get_contest_data, data_type='json',
            func_args=(link, title, contest_key),
        )
        if len(missing_keys := EXPECTED_CONTEST_DATA_KEYS - set(contest_data.keys())) > 0:
            # see if the required data is in entry_info
            still_missing = []
            for key in missing_keys:
                if key in entry_info:
                    contest_data[key] = entry_info[key]
                else:
                    still_missing.append(key)
            assert len(still_missing) == 0, \
                f"Could not find contest data for {still_missing}"

        contest_dict = {
            'contest_id': contest_key,
            'date': entry_info.date,
            'sport': entry_info.sport,
            'title': title,
            'fee': entry_info.fee,
            'entries': entry_info.entries,
            'winners': contest_data['winners'],
            'top_score': contest_data['top_score'],
            'last_winning_score': contest_data['last_winning_score'],
        }

        self._contest_dicts.append(contest_dict)

        for lineup_data in contest_data['lineups_data']:
            lineup_df = self.get_entry_lineup_df(lineup_data)
            lineup_df['contest'] = contest_key
            lineup_df['date'] = entry_info.date
            lineup_df['sport'] = entry_info.sport
            self._player_draft_dfs.append(lineup_df)

        self.processed_contests.add(contest_id)

    @abstractstaticmethod
    def get_contest_identifiers(entry_info) -> tuple[str, tuple, str, str]:
        """ returns - (contest key, contest id, entry link, browser page title) """
        raise NotImplementedError()

    def pause(self, msg=None):
        if self.interactive:
            input("Paused for '" + (msg or "?") + "' <Enter> to continue ")
            return
        pause_for = random.randint(PAUSE_MIN, PAUSE_MAX)
        msg = "" if msg is None else ": " + msg
        LOGGER.info("Pausing for %i seconds%s", pause_for, msg)
        time.sleep(pause_for)

    def browse_to(self, url, pause_before=True, title=None):
        """
        browse to the requested page, possibly with a pause before browsing
        if the current browser title is the same as title OR the current browser URL is the same as url then
        don't load anything new, use the current page content

        pause_before - if false then don't pause or prompt user to continue
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
                    self.pause(msg=f"before post login link #{i}getting url content")
                LOGGER.info("Going to post login url #%i '%s'", i, url)
                self.browser.get(url)

        if pause_before:
            self.pause("before getting url content")

        LOGGER.info("Getting content at '%s'", url)
        self.browser.get(url)

    def get_data(self, cache_key: str, func: callable, data_type: str = 'csv',
                 func_args: Optional[tuple] = None, func_kwargs: Optional[dict] = None):
        """
        get data related to the key, first try the cache, if that fails call func
        and cache the result before returning result

        type - the type of data to be loaded. csv -> dataframe, json -> dict/list, txt|html -> str.
            This should be the same as the data type returned by func
        func - a function that when executed returns the required data, takes as arguments, (link, title)
        func_args - positional arguments to pass to func
        func_kwargs - kwargs to pass to func
        """
        # see if it in the cache
        if os.sep in cache_key:
            cache_key = cache_key.replace(os.sep, "|")
        if self.cache_path is not None and \
           os.path.isfile(cache_filepath := os.path.join(self.cache_path, f"{cache_key}.{data_type}")):
            if data_type == 'csv':
                return pd.read_csv(cache_filepath)
            elif data_type == 'json':
                with open(cache_filepath, "r") as f_:
                    return json.load(f_)
            elif data_type in {'html', 'txt'}:
                with open(cache_filepath, "r") as f_:
                    return f_.read()

            raise ValueError(f"Don't know how to load '{cache_filepath}' from cache")

        data = func(
            *(func_args or tuple()),
            **(func_kwargs or {}),
        )

        if self.cache_path is not None:
            if data_type == 'csv':
                data.to_csv(cache_filepath)
            elif data_type == 'json':
                with open(cache_filepath, "w") as f_:
                    json.dump(data, f_)
            elif data_type in {'html', 'txt'}:
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
        interactive: bool = False,
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
        interactive=interactive,
    )
