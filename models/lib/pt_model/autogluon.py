import shutil
import tempfile

import pandas as pd
import psutil
import torch
from autogluon.tabular import TabularPredictor
from fantasy_py import FantasyException, InvalidArgumentsException, log
from fantasy_py.inference import PTPredictModel
from ledona import bytes2human, get_total_memory, get_used_memory

from .wrapper import PTEstimatorWrapper


class AutoGluonWrapper(PTEstimatorWrapper):
    """wrapper around autogluon, simplifies instantiation, fitting and saving the model"""

    VERSIONS_FOR_DEPS = [
        "sklearn",
        "autogluon.tabular",
        "lightgbm",
        "catboost",
        "xgboost",
        "fastai",
        "ray",
    ]

    _TARGET_COL = "autogluon_target_variable"

    predictor: TabularPredictor
    time_limit: int | None
    """training time limit in secs"""
    preset: str
    """fit preset"""
    disable_cuda: bool
    """
    if cuda is not available this is ignored, if cuda device is found then
    if this is true it will be ignored (cpu only training), if false then cuda device will be used
    """

    def __init__(
        self,
        model_filebase,
        preset: str | None = None,
        time_limit: int | None = None,
        disable_cuda: bool = False,
        **init_kwargs,
    ):
        """preset: default is medium"""
        self.disable_cuda = disable_cuda
        self._logger = log.get_logger(__name__)
        if preset is None:
            self._logger.info("Autogluon preset not set, falling back on medium")
        self.preset = preset or "medium"
        self.time_limit = time_limit
        self._model_tmpdir = tempfile.TemporaryDirectory(
            prefix=f"autogluon-model:{model_filebase}.", delete=False
        )
        free_gb = shutil.disk_usage(self._model_tmpdir.name).free / 1024**3
        if free_gb < 10:
            raise FantasyException(
                f"Insufficient disk space at '{self._model_tmpdir.name}': {free_gb:.1f} GB free, 10 GB required"
            )
        self._logger.info(
            "Autogluon temp model data will be written to '%s'", self._model_tmpdir.name
        )
        self.predictor = TabularPredictor(
            self._TARGET_COL, problem_type="regression", path=self._model_tmpdir.name, **init_kwargs
        )

    def __exit__(self, *_):
        """remove the temp dirextory when the model context manager is exited"""
        self._logger.info("Removing autogluon temp model data from '%s'", self._model_tmpdir.name)
        self._model_tmpdir.cleanup()

    @property
    def sample_weight_support(self):
        return True

    def fit(self, x: pd.DataFrame, y: pd.Series):
        if self._TARGET_COL in x:
            raise InvalidArgumentsException(
                f"the target column {self._TARGET_COL} should not already be in x"
            )
        x_with_y = x.assign(**{self._TARGET_COL: y})
        fit_kwargs: dict = {"presets": self.preset}
        if self.time_limit:
            fit_kwargs["time_limit"] = self.time_limit

        if torch.cuda.is_available():
            if self.disable_cuda:
                self._logger.info(
                    "CUDA device is available but will not be used because disable_cuda is True"
                )
                fit_kwargs["num_gpus"] = 0
            else:
                fit_kwargs["ag_args_fit"] = {"num_gpus": torch.cuda.device_count()}
                cpu_count = psutil.cpu_count(logical=True)
                if cpu_count is None:
                    self._logger.warning(
                        "Unable to determine the number of logical CPUs available for training. AutoGluon will choose"
                    )
                elif self.preset.startswith("best") or self.preset == "bq":
                    # Calculate safe cores based on available memory
                    # cpus_to_use = min(cpus - 2, floor(total_mem / 4GB))
                    total_mem = get_total_memory()
                    used_mem = get_used_memory()
                    avail_mem_gb = (total_mem - used_mem) / (1024**3)
                    # Reserve 4GB for OS/Driver, then divide by 4GB per core
                    safe_cpus = min(int(avail_mem_gb / 4) or 1, cpu_count - 2)
                    self._logger.info(
                        "Autogluon 'best' preset will use gpu and %i cpus based on avail memory of %.1fGB. "
                        "total-mem=%s used-mem=%s",
                        safe_cpus,
                        avail_mem_gb,
                        bytes2human(total_mem),
                        bytes2human(used_mem),
                    )
                    fit_kwargs["num_cpus"] = safe_cpus
                    fit_kwargs["ag_args_ensemble"] = {"fold_fitting_strategy": "parallel_local"}
                else:
                    fit_kwargs["num_cpus"] = cpu_count - 2

        self._logger.info("Autogluon fitting with kwargs: %s", fit_kwargs)
        self.predictor.fit(x_with_y, **fit_kwargs)

        self.predictor.fit_summary()
        self._logger.success("Autogluon fitted!")

    def predict(self, x: pd.DataFrame):
        with torch.device("cpu"):
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

    def get_training_desc_info(self, *args):
        """
        update the training_desc_info dict with information describing the
        trained model
        """
        info = super().get_training_desc_info(*args)

        ag_info = self.predictor.info()
        clean_info = self._model_info_value_cleanup(ag_info)
        cuda_available = torch.cuda.is_available()
        info["autogluon"] = {
            "preset": self.preset,
            "cuda_available": cuda_available,
            "cuda_disable_for_fit": self.disable_cuda,
            "cuda_used": cuda_available and not self.disable_cuda,
            "info": clean_info,
        }

        return info

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
