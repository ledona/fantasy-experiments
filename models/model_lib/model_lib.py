"""
helpful code for logging models to mlflow
"""

from abc import ABC, abstractmethod
from typing import Callable
import logging
import os

import mlflow
import pandas as pd

from fantasy_py import log, get_git_desc, UnexpectedValueError
from fantasy_py.inference import Model


_LOGGER = log.get_logger(__name__)


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


def _run_name_exists(exp_name, run_name):
    """returns true if the run_name is already in use for the experiment"""
    runs = mlflow.search_runs(
        experiment_names=[exp_name],
        filter_string=f"run_name = '{run_name}'",
    )
    return len(runs) > 0


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
        }, "mlf_tracking_uri is the only supported tracker setting"
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

    if experiment_name and run_name and _run_name_exists(experiment_name, run_name):
        raise ValueError(f"run_name '{run_name}' already exists for experiment '{experiment_name}'")

    final_run_tags = {key: str(value) for key, value in run_tags.items()}
    if "fantasy-sha" not in final_run_tags:
        final_run_tags["fantasy-sha"] = get_git_desc()

    _LOGGER.debug(
        "starting run name='%s' description='%s' tags=%s",
        run_name,
        run_description,
        run_tags,
    )
    with mlflow.start_run(
        run_name=run_name, tags=final_run_tags, description=run_description
    ) as run_:
        _LOGGER.info(
            "Training exp='%s' run='%s' tags=%s",
            experiment_name,
            run_name or run_.info.run_id,
            run_tags,
        )
        train_ret = train_func(*(train_args or []), **(train_kwargs or {}))
        model_name, metrics, artifact_paths, addl_tags = train_ret[1:]
        mlflow.set_tag("model_name", model_name)
        _LOGGER.info(
            "Trained exp='%s' run='%s' model='%s'",
            experiment_name,
            run_name or run_.info.run_id,
            model_name,
        )

        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(artifact_path)
        if addl_tags:
            mlflow.set_tags(addl_tags)

        _LOGGER.info(
            "Logging training run results for model-name='%s', metrics=%s, artifact_paths=%s",
            model_name,
            metrics,
            artifact_paths,
        )

    return (run_.info.run_id, train_ret)


def _parse_tracking_settings(tracker_settings: dict | None):
    if not tracker_settings:
        return None

    assert set(tracker_settings.keys()) <= {
        "mlf_tracking_uri"
    }, "mlf_tracking_url is the only supported tracker setting"
    mlf_tracking_uri = tracker_settings.get("mlf_tracking_uri", "local-fantasy-mlrun")
    mlflow.set_tracking_uri(mlf_tracking_uri)
    _LOGGER.debug("retrieving from mlflow URI %s", mlf_tracking_uri)
    return mlf_tracking_uri


def retrieve(
    experiment_name: str | None = None,
    run_id=None,
    model_name=None,
    active_only=True,
    tracker_settings: dict | None = None,
    dest_path: str | None = None,
    **run_tags,
) -> None | list[ModelObj]:
    """
    Retrieve the active model for the requested parameters, either provide the model name XOR
    other arguments

    active_only - ignored if run_id provided
    """
    mlf_tracking_uri = _parse_tracking_settings(tracker_settings)
    if run_id is not None:
        assert len(run_tags) == 0 and model_name is None and experiment_name is None
        run = mlflow.get_run(run_id)
        runs = [run]
    else:
        filter_strings = [
            f"tags.{tag_name} = '{tag_value}'" for tag_name, tag_value in run_tags.items()
        ]
        if model_name is not None:
            filter_strings.append(f"tags.model_name = '{model_name}'")
        if active_only:
            filter_strings.append(f"tags.active = '{True}'")
        if len(filter_strings) > 0:
            filter_string = " and ".join(filter_strings)
            _LOGGER.info(
                "Searching for mlflow runs: exp-name='%s' filter_string=\"%s\"",
                experiment_name,
                filter_string,
            )
        else:
            _LOGGER.warning(
                "No run query specified. active_only as True or run_id, model_name or "
                "kwargs for run tag filters must be provided."
            )
            filter_string = None

        runs = mlflow.search_runs(
            search_all_experiments=(experiment_name is None),
            experiment_names=[experiment_name] if experiment_name is not None else None,
            filter_string=filter_string,
            output_format="list",
        )
        _LOGGER.info("%i runs found", len(runs))

    models = []
    for i, run in enumerate(runs, 1):
        if experiment_name is None:
            exp = mlflow.get_experiment(run.info.experiment_id)
            run_exp_name = exp.name
        else:
            run_exp_name = experiment_name
        print(
            f"\nrun #{i}\nexp-name='{run_exp_name}' "
            f"model-name='{run.data.tags['model_name']}' "
            f"run-id={run.info.run_id}"
        )
        if _LOGGER.isEnabledFor(logging.DEBUG):
            print(f"{run.data.tags}")
        if dest_path is None:
            continue
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, tracking_uri=mlf_tracking_uri, dst_path=dest_path
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

    return models if dest_path else None


def activate_model(run_id, tracker_settings: dict | None = None):
    """activate the model"""
    _parse_tracking_settings(tracker_settings)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("active", f"{True}")


def deactivate_models(
    run_id=None, model_name=None, tags=None, tracker_settings: dict | None = None
):
    """
    deactive the requested model(s)
    """
    _parse_tracking_settings(tracker_settings)
    if run_id:
        assert model_name is None and tags is None
        run_ids = [run_id]
    else:
        if model_name:
            assert tags is None
            filter_strings = [f"tags.model_name = '{model_name}'"]
        else:
            assert tags is not None
            filter_strings = [
                f"tags.{tag_name} = '{tag_value}'" for tag_name, tag_value in tags.items()
            ]

        filter_strings.append(f"tags.active = '{True}'")
        filter_string = " and ".join(filter_strings)
        run_ids = [
            run.info.run_id
            for run in mlflow.search_runs(
                search_all_experiments=True,
                filter_string=filter_string,
                output_format="list",
            )
        ]

    _LOGGER.info("Deactivating %i models", len(run_ids))
    for ri in run_ids:
        with mlflow.start_run(run_id=ri):
            mlflow.set_tag("active", f"{False}")
