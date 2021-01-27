from abc import ABC, abstractmethod
from importlib import import_module
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

    def __init__(self, chrome_port=None):
        chrome_options = ChromeOptions()
        chrome_options.add_experimental_option("excludeSwitches", ['enable-automation'])

        if chrome_port is not None:
            chrome_options.add_experimental_option(
                "debuggerAddress", f"127.0.0.1:{chrome_port}"
            )

        LOGGER.info("Opening '%s'", self.SERVICE_URL)

        self.browser = webdriver.Chrome("chromedriver", options=chrome_options)
        try:
            self.browser.get(self.SERVICE_URL)
        finally:
            if chrome_port is None:
                self.browser.close()

        self.contest_history_df = None
        self.draft_history_df = None
        self.betting_history_df = None

    def wait_on_login(self):
        try:
            # if found, then signin is required
            self.browser.find_element(*self.LOC_SIGN_IN)
        except NoSuchElementException:
            LOGGER.info("Log in link %s not found. Assuming account is already logged in",
                        self.LOC_SIGN_IN)
            return

        LOGGER.info("Waiting for you to log in...")
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

    @abstractmethod
    def get_entries(self, history_file_dir, sport, start_date, end_date):
        """ return an iterator that yields entries """
        raise NotImplementedError()

    @abstractmethod
    def process_entry(self, entry_info):
        """
        process a contest entry. if the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        raise NotImplementedError()


def get_service_data_retriever(service: str, chrome_port=None) -> ServiceDataRetriever:
    """ attempt to import the data retriever for the requested service """
    module = import_module(service)
    class_ = getattr(module, service.capitalize())
    return class_(chrome_port=chrome_port)
