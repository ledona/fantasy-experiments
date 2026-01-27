from pprint import pprint

import joblib
import pandas as pd
from fantasy_py import FantasyException, log
from tpot2 import TPOTRegressor


class TPOTNotYetFittedError(FantasyException):
    """raised if an operation requires a fitted model and the regressor is not yet fitted"""


class TPOTWrapper:
    """wrapper to facilitate tpot training"""

    def __init__(self, tpot_light=False, **model_params):
        params = model_params.copy()
        if "epochs_max" in params:
            params["generations"] = params.pop("epochs_max")
        for param in list(params.keys()):
            if param.startswith("tp:"):
                params[param[3:]] = params.pop(param)

        if tpot_light:
            model_params["search_space"] = "linear-light"
        self.tpot_regressor = TPOTRegressor(**params)

    @property
    def max_generation(self):
        if self.tpot_regressor.evaluated_individuals is None:
            raise TPOTNotYetFittedError(
                "evaluated_individuals is empty. has the model been fitted?"
            )
        max_gen = self.tpot_regressor.evaluated_individuals.Generation.max()
        return max_gen

    def update_training_desc_info(self, training_desc_info):
        """update the training desc info after fitting"""
        training_desc_info["generations_tested"] = self.max_generation

    def log_fitted_model(self):
        """Log something to stdout/log describing the fitted model"""
        log.get_logger(__name__).success("TPOT fitted in %i generation(s)", self.max_generation)
        pprint(self.tpot_regressor.fitted_pipeline_)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        try:
            self.tpot_regressor.fit(x, y)
        except Exception as e:
            exception_desc = str(e)
            if (
                "argmax of an empty sequence" in exception_desc
                or "No individuals could be evaluated in the initial population" in exception_desc
            ):
                # Check how many pipelines were attempted
                n_evaluated = (
                    len(self.tpot_regressor.evaluated_individuals)
                    if self.tpot_regressor.evaluated_individuals is not None
                    else 0
                )
                log.get_logger(__name__).error(
                    f"TPOT failed to find valid a pipeline. Pipelines evaluated: {n_evaluated}. "
                    "Consider increasing time limits or simplifying config."
                )
                raise TPOTNotYetFittedError(
                    "Fitting did not produce any valid pipelines, see log for details"
                ) from e
            raise

    def predict(self, x: pd.DataFrame):
        return self.tpot_regressor.predict(x)

    def save_artifact(self, filepath_base: str) -> str:
        artifact_filepath = filepath_base + ".pkl"
        joblib.dump(self.tpot_regressor.fitted_pipeline_, artifact_filepath)
        return artifact_filepath
