import logging
import os
from typing import cast

from fantasy_py import (
    log,
    # SPORT_DB_MANAGER_DOMAIN,
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    UnexpectedValueError,
)

# register services
from fantasy_py.lineup import FantasyService

# from fantasy_py.sport import SportDBManager
from fantasy_py.inference import Model

from .model_lib import train_and_log

_LOGGER = log.get_logger(__name__)
_LOGGER.setLevel(logging.DEBUG)


def archive_model(
    model_filepath: str,
    experiment_name: str,
    experiment_description: None | str = None,
    run_name: str | None = None,
    run_description: str | None = None,
    tracker_settings: dict | None = None,
    run_tags: dict | None = None,
):
    """
    Archive the model at the filepath, useful if the model already exists
    For new training runs use do_training_run instead

    model - if None the load the model object from the filepath,
        if not None then assume this is the model for at the filepath
    """
    _LOGGER.debug("archiving model at '%s'", model_filepath)
    model = Model.load(model_filepath)
    model_name, target_stat, framework = os.path.basename(model_filepath).split(".", 3)[:3]
    assert model.name == model_name, "expecting the filepath to include the model name"
    assert model.target.name == target_stat, "expecting the target to match"

    sport = model_name.split("-", 1)[0].lower()
    # db_manager: SportDBManager = CLSRegistry.get_class(SPORT_DB_MANAGER_DOMAIN, sport)

    service_abbr = model_name.rsplit("-", 1)[-1]
    for cls in CLSRegistry.get_classes(FANTASY_SERVICE_DOMAIN):
        if cls.ABBR == service_abbr:
            break
    else:
        raise UnexpectedValueError(f"Unknown service '{service_abbr}'")

    assert model.performance, "model performance not found"
    metrics = cast(dict[str, float], model.performance.copy())
    metrics_season = metrics.pop("season")
    final_run_tags = {
        "sport": sport,
        "service": service_abbr,
        "player-team": model.target.p_or_t.name,
        "target": f"{model.target.type}:{model.target.name}",
        "metrics-seasons": metrics_season,
        "framework": framework,
        **model.parameters,
        **(run_tags or {}),
    }

    model_dir = os.path.dirname(model_filepath)
    model_artifacts = [
        art_path if os.path.isabs(art_path) else os.path.join(model_dir, art_path)
        for art_path in model.artifacts
    ] + [model_filepath]

    return train_and_log(
        experiment_name,
        (lambda: (model, model.name, metrics, model_artifacts, None)),
        experiment_description=experiment_description,
        run_name=run_name,
        run_description=run_description,
        tracker_settings=tracker_settings,
        run_tags=final_run_tags,
    )
