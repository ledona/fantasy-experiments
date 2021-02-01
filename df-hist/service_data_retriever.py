from abc import ABC, abstractmethod, abstractclassmethod
from importlib import import_module
from typing import Optional
import logging

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options as ChromeOptions

LOGGER = logging.getLogger(__name__)


class ServiceDataRetriever(ABC):
    # URL to the service home page
    SERVICE_URL: str
    # locator for the signin/login link. if present then the account is not logged in
    LOC_SIGN_IN: tuple[str, str]
    LOC_LOGGED_IN: tuple[str, str]

    # how long to wait for login before timing out
    TIMEOUT_LOGIN = 30

    def __init__(
            self, browser_address: Optional[str] = None, browser_debug_port: Optional[bool] = None,
            browser_profile_path: Optional[str] = None,
    ):
        """
        browser_address - the connection address of the chrome instance to use. e.g. ip-address:port
        """
        chrome_options = ChromeOptions()
        if browser_address is not None:
            assert browser_debug_port is None
            assert browser_profile_path is None

            LOGGER.info("try and connect to an existing browser at %s", browser_address)
            chrome_options.add_experimental_option(
                "debuggerAddress", browser_address
            )
        else:
            # try and create a new browser to run the retrieval from
            LOGGER.info("Opening new browser")
            chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])
            if browser_debug_port is not None:
                LOGGER.info("Exposing chrome debugging on port '%s'", browser_debug_port)
                chrome_options.add_argument(f"--remote-debugging-port={browser_debug_port}")
            if browser_profile_path is not None:
                LOGGER.info("Using chrome profile at '%s'", browser_profile_path)
                chrome_options.add_argument(f"--user-data-dir={browser_profile_path}")

        self.browser = webdriver.Chrome("chromedriver", options=chrome_options)

        self.contest_history_df = None
        self.draft_history_df = None
        self.betting_history_df = None

    def wait_on_login(self):
        LOGGER.info("Opening '%s'", self.SERVICE_URL)
        self.browser.get(self.SERVICE_URL)

        try:
            # if found, then signin is required
            self.browser.find_element(*self.LOC_SIGN_IN)
        except NoSuchElementException:
            LOGGER.info("Log in link %s not found. Assuming account is already logged in",
                        self.LOC_SIGN_IN)
            return

        LOGGER.info("Waiting %i seconds for you to log in...", self.TIMEOUT_LOGIN)
        WebDriverWait(self.browser, self.TIMEOUT_LOGIN).until(
            EC.presence_of_element_located(self.LOC_LOGGED_IN),
            "Still not logged in..."
        )

    def confirm_logged_in(self):
        """ use self.LOC_LOGGED_IN to confirm that the account is logged in """
        try:
            # if found, then signin is required
            self.browser.find_element(*self.LOC_LOGGED_IN)
        except NoSuchElementException:
            LOGGER.info("Logged in link %s not found! Confirmation of logged in status failed",
                        self.LOC_LOGGED_IN)
            return

    @abstractclassmethod
    def get_entries(cls, history_file_dir, sport, start_date, end_date):
        """ return an iterator that yields entries """
        raise NotImplementedError()

    @abstractmethod
    def process_entry(self, entry_info):
        """
        process a contest entry. if the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        raise NotImplementedError()


def get_service_data_retriever(
        service: str,
        browser_address: Optional[str] = None,
        browser_debug_port: Optional[str] = None,
        browser_profile_path: Optional[str] = None,
) -> ServiceDataRetriever:
    """ attempt to import the data retriever for the requested service """
    module = import_module(service)
    class_ = getattr(module, service.capitalize())
    return class_(
        browser_address=browser_address,
        browser_debug_port=browser_debug_port,
        browser_profile_path=browser_profile_path
    )
