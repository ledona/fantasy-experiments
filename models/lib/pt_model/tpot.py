from pprint import pprint

import joblib
import pandas as pd
import tpot2
from fantasy_py import FantasyException, log
from sklearn.impute import SimpleImputer
from tpot2 import TPOTRegressor


class TPOTFittingError(FantasyException):
    """raised if an operation requires a fitted model and the regressor is not yet fitted"""


class TPOTWrapper:
    """wrapper to facilitate tpot training"""

    def __init__(self, tpot_light=False, **model_params):
        params = model_params.copy()
        if "epochs_max" in params:
            params["generations"] = params.pop("epochs_max")

        tpot2_version = tpot2._version.__version__

        if "max_time_mins" in model_params and tpot2_version[:5] == "0.1.8":
            params["max_time_seconds"] = params.pop("max_time_mins") * 60
        if "tp:max_eval_time_mins" in model_params and tpot2_version[:5] == "0.1.8":
            params["max_eval_time_seconds"] = params.pop("tp:max_eval_time_mins") * 60

        for param in list(params.keys()):
            if param.startswith("tp:"):
                params[param[3:]] = params.pop(param)

        if tpot_light:
            model_params["search_space"] = "linear-light"
        self.tpot_regressor = TPOTRegressor(**params)
        self._imputer = None

    @property
    def max_generation(self):
        if self.tpot_regressor.evaluated_individuals is None:
            raise TPOTFittingError("evaluated_individuals is empty. has the model been fitted?")
        max_gen = self.tpot_regressor.evaluated_individuals.Generation.max()
        return max_gen

    def update_training_desc_info(self, training_desc_info):
        """update the training desc info after fitting"""
        training_desc_info["generations_tested"] = self.max_generation

    def log_fitted_model(self):
        """Log something to stdout/log describing the fitted model"""
        log.get_logger(__name__).success("TPOT fitted in %i generation(s)", self.max_generation)
        pprint(self.tpot_regressor.fitted_pipeline_)

    def _impute_input_data(self, x):
        if not self._imputer:
            self._imputer = SimpleImputer()
            new_x = self._imputer.fit_transform(x)
        else:
            new_x = self._imputer.transform(x)
        new_x_df = pd.DataFrame(new_x)
        new_x_df.columns = x.columns
        new_x_df.index.name = x.index.name
        return new_x_df

    def fit(self, x: pd.DataFrame, y: pd.Series):
        if y.hasnans:
            raise TPOTFittingError("y has Nans")
        if (cols_w_na := x.isna().any()).any():
            logger = log.get_logger(__name__)
            logger.info(
                "TPot training data has nan values that will be imputed. %i columns with nans: %s",
                len(cols_w_na),
                sorted(cols_w_na.index),
            )
            for col in cols_w_na.index:
                if x[col].isna().all():
                    raise TPOTFittingError(f"All values for tpot training feature '{col}' are na!")
            x = self._impute_input_data(x)
            logger.info("TPot training data imputation completed. moving on to fit")
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
                raise TPOTFittingError(
                    "Fitting did not produce any valid pipelines, see log for details"
                ) from e
            raise

    def predict(self, x: pd.DataFrame):
        if x.isna().any().any():
            x = self._impute_input_data(x)
        return self.tpot_regressor.predict(x)

    def save_artifact(self, filepath_base: str) -> str:
        artifact_filepath = filepath_base + ".pkl"
        joblib.dump(self.tpot_regressor.fitted_pipeline_, artifact_filepath)
        return artifact_filepath
