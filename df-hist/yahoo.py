import logging

from selenium.webdriver.common.by import By

from service_data_retriever import ServiceDataRetriever

LOGGER = logging.getLogger(__name__)


class Yahoo(ServiceDataRetriever):
    SERVICE_URL = "https://sports.yahoo.com/dailyfantasy/"
    LOC_SIGN_IN = (By.LINK_TEXT, "Sign in")
    LOC_LOGGED_IN = (By.NAME, "Profile Picture")

    def get_entries(self, history_file_dir, sport, start_date, end_date):
        """ return an iterator that yields entries """
        raise NotImplementedError()

    def process_entry(self, entry_info):
        """
        process a contest entry. if the contest has not yet been processed then add contest
        information to the contest dataframe and draft information from non entry lineups
        """
        raise NotImplementedError()
