import joblib
import pandas as pd
from fantasy_py import log
from sklearn.dummy import DummyRegressor

from .wrapper import PTEstimatorWrapper


class DummyWrapper(PTEstimatorWrapper):
    VERSIONS_FOR_DEPS = ["sklearn"]

    def __init__(self, params):
        self._regressor = DummyRegressor(strategy=params.get("dmy:strategy"))

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self._regressor.fit(x, y)
        log.get_logger(__name__).success("Dummy fitted!")

    def predict(self, x: pd.DataFrame):
        return self._regressor.predict(x)

    def save_artifact(self, filepath_base: str):
        artifact_filepath = filepath_base + ".pkl"
        joblib.dump(self._regressor, artifact_filepath)
        return artifact_filepath
