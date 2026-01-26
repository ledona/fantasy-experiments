import os
import shutil
from tempfile import gettempdir

import pandas as pd
from autogluon.tabular import TabularPredictor
from fantasy_py import log, InvalidArgumentsException


class AutoGluonWrapper:
    """wrapper around autogluon, simplifies instantiation, fitting and saving the model"""

    _TARGET_COL = "autogluon_target_variable"
    _AUTOGLUON_PATH = os.path.join(gettempdir(), "fantasy-autogluon")

    predictor: TabularPredictor
    time_limit: int | None
    """training time limit in secs"""
    preset: str | None
    """fit preset"""

    def __init__(self, preset: str | None = None, time_limit: int | None = None, **init_kwargs):
        self.preset = preset
        self.time_limit = time_limit
        self.predictor = TabularPredictor(
            self._TARGET_COL, problem_type="regression", path=self._AUTOGLUON_PATH, **init_kwargs
        )

    def fit(self, x: pd.DataFrame, y: pd.Series):
        if self._TARGET_COL in x:
            raise InvalidArgumentsException(
                f"the target column {self._TARGET_COL} should not already be in x"
            )
        x_with_y = x.assign(**{self._TARGET_COL: y})
        fit_kwargs = {}
        if self.preset:
            fit_kwargs["presets"] = self.preset
        if self.time_limit:
            fit_kwargs["time_limit"] = self.time_limit
        self.predictor.fit(x_with_y, **fit_kwargs)
        return self

    def predict(self, x: pd.DataFrame):
        return self.predictor.predict(x)

    _MODEL_INFO_MAX_DEPTH = 5

    @classmethod
    def _model_info_value_cleanup(cls, dict_, depth=0):
        """traverse the mode info dict, any value that is not a string|number convert to a string"""
        for k, v in dict_.items():
            if v is None or isinstance(v, (int, float, str, list)):
                continue
            if not isinstance(v, dict):
                dict_[k] = str(v)
                continue

            # its a dict
            if len(v) == 0:
                continue
            if depth == cls._MODEL_INFO_MAX_DEPTH:
                dict_[k] = str(v)
                continue
            cls._model_info_value_cleanup(v, depth + 1)

    def update_training_desc_info(self, info: dict):
        """
        update the training_desc_info dict with information describing the
        trained model
        """
        ag_info = self.predictor.info()
        model_info = self.predictor.model_info(self.predictor.model_best)
        clean_model_info = self._model_info_value_cleanup(model_info)
        info["autogluon"] = {
            "version": ag_info["version"],
            "preset": self.preset,
            "best_model": self.predictor.model_best,
            "model_info": clean_model_info,
        }

    def save_artifact(self, filepath_base: str) -> str:
        """save the artifact to the requested location, return the full artifact path"""
        ag_model_pkl_path = os.path.join(
            self.predictor.path, "models", self.predictor.model_best, "model.pkl"
        )
        if not os.path.isfile(ag_model_pkl_path):
            raise FileNotFoundError(
                f"autogluon model artifact not found at the expected path of {ag_model_pkl_path}"
            )

        dest_path = filepath_base + ".pkl"
        shutil.copyfile(ag_model_pkl_path, dest_path)
        log.get_logger(__name__).info(
            "Artifact file successfully copied: '%s' -> '%s'", ag_model_pkl_path, dest_path
        )
        return dest_path

    def log_fitted_model(self):
        """Log something to stdout/log describing the fitted model"""
        log.get_logger(__name__).success(
            "Autogluon fitted. best-model:%s", self.predictor.model_best
        )
