from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod
from importlib import import_module
import functools
import json
import time
import random
from typing import Literal, Optional, Union
import logging
import os

import pandas as pd
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
import tqdm

LOGGER = logging.getLogger(__name__)

PAUSE_MIN = 3
PAUSE_MAX = 15

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
    'contest_id', 'date', 'sport', 'title', 'fee', 'entries', 'winnings',
}

# columns that will be in the player draft percentage dataframe
EXPECTED_DRAFT_PLAYER_COLS = {
    'contest', 'date', 'sport', 'team_abbr', 'position',
    'name', 'draft_pct'
}


EXPECTED_CONTEST_DATA_KEYS = {
    'lineups_data',  # list of lineup data strings
    'winners',       # number of contest winners
    'top_score',
    'last_winning_score',
}


# the return type for get_data. tuple(data, retrieved from, cache filename)
GetDataResult = tuple[Union[dict, list, pd.DataFrame, str], Literal['cache', 'web'], Optional[str]]


class WebLimitReached(Exception):
    """ raised when the web retrieval limit is reached and data must be retrieved from the web """


class ServiceDataRetriever(ABC):
    SERVICE_ABBR: str
    # URL to the service home page
    SERVICE_URL: str
    # links to go to after logging in (look like a human)
    POST_LOGIN_URLS: list[str] = []
    # locator for the signin/login link. if present then the account is not logged in
    LOC_SIGN_IN: tuple[str, str]
    LOC_LOGGED_IN: tuple[str, str]

    # how long to wait for login before timing out
    LOGIN_TIMEOUT = 300

    WAIT_TIMEOUT = 300

    def __init__(
            self, browser_address: Optional[str] = None, browser_debug_port: Optional[bool] = None,
            browser_profile_path: Optional[str] = None,
            cache_path: Optional[str] = None, cache_overwrite = False,
            interactive = False, web_limit = None,
    ):
        """
        browser_address - the connection address of the chrome instance to use. e.g. ip-address:port
        interactive - if true require user confirmation prior to every browser action
        web_limit - halt processing if this number of web retrievals is exceeded
        """
        self.cache_path = cache_path
        self.cache_overwrite = cache_overwrite
        self.interactive = interactive

        self.browser_address = browser_address
        self.browser_debug_port = browser_debug_port
        self.browser_profile_path = browser_profile_path

        self._player_draft_dfs: list[pd.DataFrame] = []
        self._contest_dicts: list[dict] = []
        self._entry_dicts: list[dict] = []
        # used to keep track of the contests that have been processed
        self.processed_contests = set()

        # count of where data was retrieved from for the entries that were processed
        self.processed_counts_by_src = {'cache': 0, 'web': 0}

        self.web_limit = web_limit

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
            LOGGER.warning("Error updating browser webdriver property", exc_info=ex)
        LOGGER.info("Connected to browser")
        return browser_

    @property
    def player_draft_df(self):
        """
        returns - dataframe with columns in EXPECTED_DRAFT_PLAYER_COLS
        """
        df = pd.concat(self._player_draft_dfs, ignore_index=True) \
               .drop_duplicates(ignore_index=True)
        assert len(missing := EXPECTED_DRAFT_PLAYER_COLS - set(df.columns)) == 0, \
            f"Missing columns: {missing}"
        return df

    @property
    def contest_df(self):
        """
        returns - dataframe with columns matching EXPECTED_CONTEST_COLS
        """
        df = pd.DataFrame(self._contest_dicts)
        assert len(missing := EXPECTED_CONTEST_COLS - set(df.columns)) == 0, \
            f"Missing columns: {missing}"
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
                input(f"About to retrieve {self.SERVICE_URL}. <Enter> to continue:")
            self.browser.get(self.SERVICE_URL)

        try:
            # if found, then signin is required
            self.browser.find_element(*self.LOC_SIGN_IN)
        except NoSuchElementException:
            LOGGER.info("Log in link %s not found. Assuming account is already logged in",
                        self.LOC_SIGN_IN)
            return

        if self.interactive:
            input("Waiting for you to log in. <Enter> to continue:")

        LOGGER.info("Waiting %i seconds for you to log in...", self.LOGIN_TIMEOUT)
        WebDriverWait(self.browser, self.LOGIN_TIMEOUT).until(
            EC.presence_of_element_located(self.LOC_LOGGED_IN),
            "Still not logged in..."
        )

    @property
    def logged_in_to_service(self) -> bool:
        """ use self.LOC_LOGGED_IN to confirm that the account is logged in """
        try:
            LOGGER.debug("Looking for logged in indication...")
            # if found, then signin has already taken place
            self.browser.find_element(*self.LOC_LOGGED_IN)
            return True
        except NoSuchElementException:
            LOGGER.error(
                "Logged in indicator %s not found! Confirmation of logged in status failed",
                 self.LOC_LOGGED_IN
            )
            return False

    @abstractclassmethod
    def get_historic_entries_df_from_file(cls, history_file_dir):
        """
        return a dataframe with entries data retrieved from the service's entry history data files
            The dataframe must include all columns named in EXPECTED_HISTORIC_ENTRIES_DF_COLS.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_entry_lineup_data(self, link, title):
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
    def get_contest_data(self, link, contest_key, entry_info) -> dict:
        """
        get all contest data that is not in the entry info and return in a dict with keys
        in EXPECTED_CONTEST_DATA_KEYS
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
        if entry_dict['link'] is None:
            entry_dict['link'] = self.get_entry_link(entry_info)

        self._entry_dicts.append(entry_dict)

        contest_key = f"{self.SERVICE_ABBR}-{entry_info.sport}-{entry_info.date:%Y%m%d}-{entry_info.title}"
        contest_id =  (entry_info.sport, entry_info.date, entry_info.title)
        entry_key = f"{contest_key}-entry-{entry_info.entry_id}"

        # handle entry lineup info
        entry_lineup_data, entry_src, _ = self.get_data(
            entry_key, self.get_entry_lineup_data, data_type='html',
            func_args=(entry_dict['link'], entry_info.title),
        )
        LOGGER.info(
            "Entry lineup for '%s' retrieved from %s",
            entry_key, entry_src
        )
        entry_lineup_df = self.get_entry_lineup_df(entry_lineup_data)
        entry_lineup_df['contest'] = contest_key
        entry_lineup_df['date'] = entry_info.date
        entry_lineup_df['sport'] = entry_info.sport
        self._player_draft_dfs.append(entry_lineup_df)

        # if contest has been processed then we are done
        if contest_id in self.processed_contests:
            LOGGER.info("Contest data for '%s' already processed. Skipping to next entry.", contest_id)
            return

        # process contest data
        contest_data, contest_src, _ = self.get_data(
            contest_key, self.get_contest_data, data_type='json',
            func_args=(entry_dict['link'], contest_key, entry_info),
        )

        src = 'web' if 'web' in (contest_src, entry_src) else 'cache'
        self.processed_counts_by_src[src] += 1
        LOGGER.info(
            "Contest data for '%s' retrieved from %s",
            contest_key, contest_src
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
            'title': entry_info.title,
            'fee': entry_info.fee,
            'entries': entry_info.entries,
            'winners': contest_data['winners'],
            'top_score': contest_data['top_score'],
            'last_winning_score': contest_data['last_winning_score'],
        }

        self._contest_dicts.append(contest_dict)

        for lineup_data in contest_data['lineups_data']:
            lineup_df = self.get_opp_lineup_df(lineup_data)
            lineup_df['contest'] = contest_key
            lineup_df['date'] = entry_info.date
            lineup_df['sport'] = entry_info.sport
            self._player_draft_dfs.append(lineup_df)

        self.processed_contests.add(contest_id)

    def get_opp_lineup_df(self, lineup_data):
        """ default to using get_entry_lineup_df """
        return self.get_entry_lineup_df(lineup_data)

    @abstractstaticmethod
    def get_entry_link(entry_info) -> str:
        raise NotImplementedError()

    def pause(self, msg=None, pause_min=PAUSE_MIN, pause_max=PAUSE_MAX, progress_bar=True):
        """
        Pause for a random amount of time

        msg - message to print describing the pause
        pause_min - minimum pause duration
        pause_max - maximum pause duration
        progress_bar - show a progress bar counting down the pause.
           if final pause duration is <= 2 then no progress bar is displayed regardless
        """
        if self.interactive:
            input("Paused for '" + (msg or "?") + "' <Enter> to continue:")
            return
        assert pause_min <= pause_max
        pause_for = random.randint(pause_min, pause_max)
        if msg is None:
            msg = "?"
        if pause_for > 2 and progress_bar:
            for _ in tqdm.trange(pause_for, unit="", desc=f"pause for '{msg}'", leave=False):
                time.sleep(.995)
        else:
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
            for i, post_login_url in enumerate(self.POST_LOGIN_URLS, 1):
                if pause_before:
                    self.pause(msg=f"wait before post login link #{i}")
                LOGGER.info("Going to post login url #%i '%s'", i, post_login_url)
                self.browser.get(post_login_url)

        if pause_before:
            self.pause(f"wait before url '{url}'")

        LOGGER.info("Getting content at '%s'", url)
        self.browser.get(url)

    def get_data(
        self, cache_key: str, func: callable, data_type: str = 'csv',
        func_args: Optional[tuple] = None, func_kwargs: Optional[dict] = None
    ) -> GetDataResult:
        """
        get data related to the key, first try the cache, if that fails call func
        and cache the result before returning result

        type - the type of data to be loaded. csv -> dataframe, json -> dict/list, txt|html -> str.
            This should be the same as the data type returned by func
        func - a function that when executed returns the required data, takes as arguments, (link, title)
        func_args - positional arguments to pass to func
        func_kwargs - kwargs to pass to func
        """
        cache_filepath = None

        # see if it in the cache
        if os.sep in cache_key:
            cache_key = cache_key.replace(os.sep, "|")
        if self.cache_path is not None and \
           os.path.isfile(cache_filepath := os.path.join(self.cache_path, f"{cache_key}.{data_type}")) and \
           not self.cache_overwrite:
            if data_type == 'csv':
                return pd.read_csv(cache_filepath), 'cache', cache_filepath
            elif data_type == 'json':
                with open(cache_filepath, "r") as f_:
                    return json.load(f_), 'cache', cache_filepath
            elif data_type in {'html', 'txt'}:
                with open(cache_filepath, "r") as f_:
                    return f_.read(), 'cache', cache_filepath

            raise ValueError(f"Don't know how to load '{cache_filepath}' from cache")

        if self.web_limit is not None and \
           self.web_limit < self.processed_counts_by_src['web']:
            LOGGER.info("Data not in cache and web retrieval limit reached. Processing stopped on %s", cache_key)
            raise WebLimitReached(cache_key)

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

        return data, 'web', cache_filepath

def get_service_data_retriever(
        service: str,
        cache_path: Optional[str] = None,
        cache_overwrite = False,
        browser_address: Optional[str] = None,
        browser_debug_port: Optional[str] = None,
        browser_profile_path: Optional[str] = None,
        interactive: bool = False,
        web_limit: Optional[int] = None,
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
        cache_overwrite=cache_overwrite,
        interactive=interactive,
        web_limit=web_limit,
    )
