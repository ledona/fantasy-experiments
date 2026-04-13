import joblib
import pandas as pd
from fantasy_py import log
from fantasy_py.inference.flaml import FlamlModel
from flaml import AutoML

from .wrapper import PTEstimatorWrapper


class FlamlWrapper(PTEstimatorWrapper):
    """wrapper around flaml AutoML for regression"""

    VERSIONS_FOR_DEPS = ["flaml", "sklearn"]

    def __init__(
        self,
        time_budget: int | None = None,
        n_jobs: int | None = None,
        use_gpu: bool = False,
        concurrent_trials: int | None = None,
        sample_weight: str | None = None,
    ):
        """
        time_budget: max training time in seconds
        n_jobs: number of parallel threads (-1 = all available)
        use_gpu: enable GPU use (gpu_per_trial=1)
        concurrent_trials: number of concurrent trials, requires Ray
        sample_weight: column name in x containing per-sample training weights
        """
        self._logger = log.get_logger(__name__)
        fit_kwargs: dict = {"task": "regression"}
        if time_budget is not None:
            fit_kwargs["time_budget"] = time_budget
        if n_jobs is not None:
            fit_kwargs["n_jobs"] = n_jobs
        if use_gpu:
            fit_kwargs["gpu_per_trial"] = 1
        if concurrent_trials is not None:
            fit_kwargs["n_concurrent_trials"] = concurrent_trials

        self._fit_kwargs = fit_kwargs
        self._sample_weight_col = sample_weight
        self._regressor = AutoML()

    @property
    def sample_weight_support(self):
        return True

    def fit(self, x: pd.DataFrame, y: pd.Series):
        fit_kwargs = self._fit_kwargs.copy()
        if self._sample_weight_col and self._sample_weight_col in x:
            fit_kwargs["sample_weight"] = x[self._sample_weight_col].to_numpy()
            x = x.drop(columns=self._sample_weight_col)
        self._regressor.fit(FlamlModel.sanitize_columns(x), y, **fit_kwargs)
        self._logger.success("FLAML fitted!")

    def predict(self, x: pd.DataFrame):
        return self._regressor.predict(FlamlModel.sanitize_columns(x))

    def save_artifact(self, filepath_base: str) -> str:
        artifact_filepath = filepath_base + ".pkl"
        joblib.dump(self._regressor, artifact_filepath)
        return artifact_filepath

    def get_training_desc_info(self, *args):
        info = super().get_training_desc_info(*args)
        info["flaml"] = {
            "best_estimator": self._regressor.best_estimator,
            "best_config": self._regressor.best_config,
            "best_loss": self._regressor.best_loss,
        }
        return info
