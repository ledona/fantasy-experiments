import os
from tempfile import gettempdir

import pandas as pd
import torch
from fantasy_py import InvalidArgumentsException, dt_to_filename_str, log
from fantasy_py.inference import NNRegressor


class NNWrapper:
    """wrapper to help train a NN, can consume and translate the args defined in cfg"""

    def __init__(self, x_test: pd.DataFrame, y_test: pd.Series, model_filebase, **params):
        self._fitted = None
        self._logger = log.get_logger(__name__)
        self.x_test = x_test
        self.y_test = y_test

        model_params = dict(params)
        input_size = len(x_test.columns)
        resume_filepath = (
            model_params.pop("nn:resume_checkpoint_filepath")
            if "nn:resume_checkpoint_filepath" in model_params
            else None
        )
        if resume_filepath is not None:
            model, best_model_info, optimizer_state = NNRegressor.load_checkpoint(
                resume_filepath, input_size, **model_params
            )

            assert model.checkpoint_dir is not None
            if not os.path.isdir(model.checkpoint_dir):
                raise FileNotFoundError(
                    f"Checkpoint model's checkpoint dir '{model.checkpoint_dir}' "
                    "is not a valid directory"
                )
            self.fit_kwargs = {
                "resume_from_checkpoint": True,
                "resume_best_model_info": best_model_info,
                "resume_optimizer_state": optimizer_state,
            }
        else:
            self.fit_kwargs = None
            if model_params.get("checkpoint_dir") is None:
                default_checkpoint_root_dir = os.path.join(gettempdir(), "fantasy-nn-checkpoints")
                if not os.path.isdir(default_checkpoint_root_dir):
                    self._logger.info(
                        "Creating default checkpoint root dir '%s'", default_checkpoint_root_dir
                    )
                    os.mkdir(default_checkpoint_root_dir)
                model_params["nn:checkpoint_dir"] = os.path.join(
                    gettempdir(),
                    "fantasy-nn-checkpoints",
                    model_filebase + "-" + dt_to_filename_str(),
                )

            if os.path.isfile(model_params["nn:checkpoint_dir"]):
                raise InvalidArgumentsException(
                    "The checkpoint directory that would be used at "
                    f"'{model_params['nn:checkpoint_dir']}' is a file! "
                    "Delete it or use another directory"
                )
            if not os.path.exists(model_params["nn:checkpoint_dir"]):
                self._logger.info(
                    "Creating nn checkpoint directory '%s'", model_params["nn:checkpoint_dir"]
                )
                os.mkdir(model_params["nn:checkpoint_dir"])

            def nn_param_name(name):
                if name.startswith("nn:"):
                    return name[3:]
                if name == "early_stop":
                    return "early_stop_epochs"
                return name

            nn_params = {nn_param_name(param): value for param, value in model_params.items()}
            model = NNRegressor(input_size, **nn_params)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self._fitted = self.model.fit(x, y, self.x_test, self.y_test, **(self.fit_kwargs or {}))
        return self

    def predict(self, x: pd.DataFrame):
        if self._fitted is None:
            raise ValueError("model not yet fitted")
        return self._fitted.predict(x)

    @property
    def epochs_trained(self):
        if self._fitted is None:
            raise ValueError("model not yet fitted")
        return self._fitted.epochs_trained

    def save(self, artifact_filepath):
        if self._fitted is None:
            raise ValueError("model not yet fitted")
        torch.save(self._fitted, artifact_filepath)
