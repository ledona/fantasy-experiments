from abc import ABC, abstractmethod
from importlib import import_module
import logging

from selenium import webdriver


LOGGER = logging.getLogger(__name__)


class ServiceDataRetriever(ABC):
    SERVICE_URL: str

    def __init__(self):
        LOGGER.info("Opening '%s'", self.SERVICE_URL)
        self.browser = webdriver.Chrome("chromedriver")
        try:
            self.browser.get(self.SERVICE_URL)
        finally:
            self.browser.close()

        self.contest_history_df = None
        self.draft_history_df = None
        self.betting_history_df = None

    @abstractmethod
    def login(self, username, password):
        raise NotImplementedError()

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


def get_service_data_retriever(service: str) -> ServiceDataRetriever:
    """ attempt to import the data retriever for the requested service """
    module = import_module(service)
    class_ = getattr(module, service.capitalize())
    return class_()
