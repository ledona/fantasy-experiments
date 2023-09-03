"""
helpful code for logging models to mlflow
"""

from abc import ABC, abstractmethod
from typing import Callable
import logging

import mlflow
import pandas as pd

from fantasy_py import log, get_git_desc


_LOGGER = log.get_logger(__name__)
_LOGGER.setLevel(logging.DEBUG)


class ModelObj(ABC):
    """abstract obj def for a trained model"""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """predict function"""
        pass


TrainFuncRetType = tuple[ModelObj, str, dict[str, float], list[str], dict[str, str]]
"""
return tuple for training func:
[sklearn compatible model, model name, metrics, artifact paths, addl run tags]
"""

TrainFuncType = Callable[..., TrainFuncRetType]
"""
type of training func arg sent to train_and_log
"""


def train_and_log(
    experiment_name: str,
    train_func: TrainFuncType,
    train_args=None,
    train_kwargs=None,
    experiment_description: str | None = None,
    tracker_settings: dict | None = None,
    run_name: None | str = None,
    run_description: None | str = None,
    run_tags=None,
) -> TrainFuncRetType:
    """
    execute a training run using the train_func and log the results and artifacts

    train_func - function that returns a Model object and a dict containing metrics
    train_args - positional args to pass to train_func
    train_kwargs - keyword args to pass to train_func
    run_[name|tags|description]
    """
    if tracker_settings:
        assert set(tracker_settings.keys()) <= {
            "mlf_tracking_uri"
        }, "mlf_tracking_url is the only supported tracker setting"
        mlf_tracking_url = tracker_settings.get("mlf_tracking_url", "local")
        _LOGGER.debug("archiving to mlflow URI %s", mlf_tracking_uri)
        mlflow.set_tracking_uri(mlf_tracking_uri)

    assert not (
        experiment_descrition and not experiment_name
    ), "experiment description without an experiment name"
    if experiment_name:
        _LOGGER.debug("Setting mlflow experiment to '%s'", experiment_name)
        mlflow.set_experiment(experiment_name)

    final_run_tags = (
        run_tags
        if "fantasy-sha" in run_tags
        else {"fantasy-sha": get_git_desc(), **run_tags}
    )

    _LOGGER.debug(
        "starting run name='%s' description='%s' tags=%s",
        run_name,
        run_description,
        run_tags,
    )
    with mlflow.start_run(
        run_name=run_name, tags=final_run_tags, description=run_description
    ):
        _LOGGER.debug("running train func")
        train_ret = train_func(*(train_args or []), **(train_kwargs or {}))
        model_name, metrics, artifact_paths, addl_tags = train_ret[1:]

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(artifact_paths)
        if addl_tags:
            mlflow.set_tags(addl_tags)

        _LOGGER.info(
            "Logging training run results for model-name='%s', metrics=%s, addl_tags=%s, artifact_paths=%s",
            model_name,
            metrics,
            addl_tags,
            artifact_paths,
        )

    return train_ret


def retrieve(
    model_name=None,
    sport=None,
    player_or_team=None,
    target=None,
    service=None,
    active_only=True,
    tracker_settings: dict | None = None,
) -> list[ModelObj]:
    """
    Retrieve the active model for the requested parameters, either provide the model name XOR
    other arguments
    """
    if model_name is not None:
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    raise NotImplementedError()
