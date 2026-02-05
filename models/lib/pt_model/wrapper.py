import importlib
import sys
from abc import ABC, abstractmethod

import pandas as pd
from fantasy_py import log, now


class PTEstimatorWrapper(ABC):
    """regression training wrapper protocol"""

    VERSIONS_FOR_DEPS: list[str] | None = None
    """
    versions for these dependencies will be added to the training desc info dict.
    Each string will be imported and the __version__ attribute will be added to the 
    dict returned by get_training_desc_info
    """

    def get_training_desc_info(
        self, dt_trained, train_cases: int, test_cases: int, validation_cases: int
    ):
        """update the training desc info after fitting"""
        info = {
            "time_to_fit": str(now() - dt_trained),
            "n_train_cases": train_cases,
            "n_test_cases": test_cases,
            "n_validation_cases": validation_cases,
        }

        versions = {"python": sys.version}
        if self.VERSIONS_FOR_DEPS:
            for dep in self.VERSIONS_FOR_DEPS:
                try:
                    module = importlib.import_module(dep)
                except ModuleNotFoundError:
                    log.get_logger(__name__).error(
                        "Version reporting dependency '%s' not available. "
                        "Remove it from VERSIONS_FOR_DEPS if it is not a valid dependency.",
                        dep,
                    )
                    versions[dep] = "module not found"
                    continue

                versions[dep] = getattr(module, "__version__", "__version__ attribute not defined")
        info["versions"] = versions

        return info

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, x: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def save_artifact(self, filepath_base: str) -> str:
        raise NotImplementedError()
