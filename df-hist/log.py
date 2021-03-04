import logging

import tqdm

LOGGING_FORMAT = '%(asctime)s-%(levelname)s-%(name)s(%(lineno)s)-%(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class TqdmStream:
    """
    https://github.com/tqdm/tqdm/issues/313
    output stream ensuring that logging and tqdm cooperate
    """
    @classmethod
    def write(cls, msg):
        tqdm.tqdm.write(msg, end='')


def setup(level=logging.INFO):
    handler = logging.StreamHandler(TqdmStream)
    handler.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT,
                                           datefmt=DATE_FORMAT))
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if root_logger.hasHandlers():
        # remove all handlers
        root_logger.handlers = []

    root_logger.addHandler(handler)
