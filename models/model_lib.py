"""
helpful code for logging models to mlflow
"""

from abc import ABC, abstractmethod
from typing import Callable
import logging
import os

import mlflow
import pandas as pd

from fantasy_py import log, get_git_desc, UnexpectedValueError, InvalidArgumentsException
from fantasy_py.inference import Model


_LOGGER = log.get_logger(__name__)
_LOGGER.setLevel(logging.DEBUG)


class ModelObj(ABC):
    """abstract obj def for a trained model"""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """predict function"""

    @property
    @abstractmethod
    def name(self) -> str:
        """model name"""


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
) -> tuple[str, TrainFuncRetType]:
    """
    execute a training run using the train_func and log the results and artifacts

    train_func - function that returns a Model object and a dict containing metrics
    train_args - positional args to pass to train_func
    train_kwargs - keyword args to pass to train_func
    run_[name|tags|description]

    returns tuple[run identifier, training return object]
    """
    if tracker_settings:
        assert set(tracker_settings.keys()) <= {
            "mlf_tracking_uri"
        }, "mlf_tracking_url is the only supported tracker setting"
        mlf_tracking_uri = tracker_settings.get("mlf_tracking_uri", "local-fantasy-mlrun")
        _LOGGER.debug("archiving to mlflow URI %s", mlf_tracking_uri)
        mlflow.set_tracking_uri(mlf_tracking_uri)

    assert not (
        experiment_description and not experiment_name
    ), "experiment description without an experiment name"
    if experiment_name:
        _LOGGER.debug("Setting mlflow experiment to '%s'", experiment_name)
        mlflow.set_experiment(experiment_name)
        if experiment_description:
            mlflow.set_experiment_tag("description", experiment_description)

    final_run_tags = (
        run_tags if "fantasy-sha" in run_tags else {"fantasy-sha": get_git_desc(), **run_tags}
    )

    _LOGGER.debug(
        "starting run name='%s' description='%s' tags=%s",
        run_name,
        run_description,
        run_tags,
    )
    with mlflow.start_run(
        run_name=run_name, tags=final_run_tags, description=run_description
    ) as run_:
        _LOGGER.debug("running train func")
        train_ret = train_func(*(train_args or []), **(train_kwargs or {}))
        model_name, metrics, artifact_paths, addl_tags = train_ret[1:]

        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(artifact_path)
        if addl_tags:
            mlflow.set_tags(addl_tags)

        _LOGGER.info(
            "Logging training run results for model-name='%s', metrics=%s, addl_tags=%s, artifact_paths=%s",
            model_name,
            metrics,
            addl_tags,
            artifact_paths,
        )

    return (run_.info.run_id, train_ret)


def retrieve(
    run_id=None, model_name=None, active_only=True, tracker_settings: dict | None = None, **run_tags
) -> list[ModelObj]:
    """
    Retrieve the active model for the requested parameters, either provide the model name XOR
    other arguments

    active_only - ignored if run_id provided
    """
    if tracker_settings:
        assert set(tracker_settings.keys()) <= {
            "mlf_tracking_uri"
        }, "mlf_tracking_url is the only supported tracker setting"
        mlf_tracking_uri = tracker_settings.get("mlf_tracking_uri", "local-fantasy-mlrun")
        _LOGGER.debug("retriving from mlflow URI %s", mlf_tracking_uri)
    else:
        mlf_tracking_uri = None

    if run_id is not None:
        assert len(run_tags) == 0 and model_name is None
        run_ids = [run_id]
    else:
        filter_strings = [
            f"tags.{tag_name} = '{tag_value}'" for tag_name, tag_value in run_tags.items()
        ]
        if model_name is not None:
            filter_strings.append(f"tags.model_name = '{model_name}'")
        if active_only:
            filter_strings.append(f"tags.active = '{True}'")
        if len(filter_strings) == 0:
            raise InvalidArgumentsException(
                "No run query specified. active_only as True or run_id, model_name or kwargs for run tag filters must be provided."
            )
        filter_string = " and ".join(filter_strings)
        run_ids = [
            run.info.run_id
            for run in mlflow.search_runs(search_all_experiments=True, filter_string=filter_string)
        ]

    models = []
    for ri in run_ids:
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=ri, tracking_uri=mlf_tracking_uri
        )
        if os.path.isfile(artifact_path):
            if not artifact_path.endswith(".model"):
                raise UnexpectedValueError(
                    f"artifact filepath is not for a model file. filepath={artifact_path}"
                )
            models.append(Model.load(artifact_path))
            continue
        for art_filepath in os.listdir(artifact_path):
            if art_filepath.endswith(".model"):
                models.append(Model.load(os.path.join(artifact_path, art_filepath)))
                break
        else:
            raise UnexpectedValueError("Model file not found in downloaded artifacts")

    return models
