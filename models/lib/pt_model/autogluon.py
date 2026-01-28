import tempfile

import pandas as pd
from autogluon.tabular import TabularPredictor
from fantasy_py import InvalidArgumentsException, log
from fantasy_py.inference import PTPredictModel


class AutoGluonWrapper:
    """wrapper around autogluon, simplifies instantiation, fitting and saving the model"""

    _TARGET_COL = "autogluon_target_variable"

    predictor: TabularPredictor
    time_limit: int | None
    """training time limit in secs"""
    preset: str | None
    """fit preset"""

    def __init__(
        self,
        model_filebase,
        preset: str | None = None,
        time_limit: int | None = None,
        **init_kwargs,
    ):
        self.preset = preset
        self.time_limit = time_limit
        model_path = tempfile.TemporaryDirectory(
            prefix=f"autogluon-model:{model_filebase}.", delete=False
        )
        self.predictor = TabularPredictor(
            self._TARGET_COL, problem_type="regression", path=model_path.name, **init_kwargs
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
        clean_info = self._model_info_value_cleanup(ag_info)
        info["autogluon"] = {"preset": self.preset, "info": clean_info}

    def save_artifact(self, filepath_base: str) -> str:
        """save the artifact to the requested location, return the full artifact path"""
        dest_path = filepath_base + ".ag"

        # ideally we would clone to the destination path, unfortunately that fails when
        # Python can't replicate filesystem metadata and permissions to the new files and directories.
        # This is the case when cloning to a mounted folder that doesn't support such operations.
        # So here we clone to a temporary location (cloning will create a cleaned up model directory),
        # then iteratively create directories and copy files to the ultimate destination.
        with tempfile.TemporaryDirectory(prefix="autogluon-model-clone-4-deploy") as tmpdir:
            local_clone = f"{tmpdir}/model_clone"
            self.predictor.clone_for_deployment(path=local_clone)
            PTPredictModel.copy_artifact_dir(local_clone, dest_path)

        return dest_path

    def log_fitted_model(self):
        """Log something to stdout/log describing the fitted model"""
        self.predictor.fit_summary()
        log.get_logger(__name__).success("Autogluon fitted!")
