import logging
import os
from typing import cast
from argparse import ArgumentParser

from fantasy_py import (
    log,
    SPORT_DB_MANAGER_DOMAIN,
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    UnexpectedValueError,
)

# register services
from fantasy_py.lineup import FantasyService

# from fantasy_py.sport import SportDBManager
from fantasy_py.inference import Model

from .model_lib import train_and_log, deactivate_models, activate_model, retrieve

_LOGGER = log.get_logger(__name__)
_LOGGER.setLevel(logging.DEBUG)


def archive_model(
    model_filepath: str,
    experiment_name: str,
    experiment_description: None | str = None,
    run_name: str | None = None,
    run_description: str | None = None,
    tracker_settings: dict | None = None,
    set_active=False,
    run_tags: dict | None = None,
):
    """
    Archive the model at the filepath, useful if the model already exists
    For new training runs use do_training_run instead

    model - if None the load the model object from the filepath,
        if not None then assume this is the model for at the filepath
    """
    _LOGGER.info("archiving model at '%s'", model_filepath)
    model = Model.load(model_filepath)
    model_name, target_stat, framework = os.path.basename(model_filepath).split(".", 3)[:3]
    assert model.name == model_name, "expecting the filepath to include the model name"
    assert model.target.name == target_stat, "expecting the target to match"

    final_run_name = run_name or f"{model.name} {model.dt_trained.strftime('%Y%m%d %H:%M:%S')}"

    sport = model_name.split("-", 1)[0].lower()
    if sport not in CLSRegistry.get_names(SPORT_DB_MANAGER_DOMAIN):
        UnexpectedValueError(
            f"Failed to parse a known sport name from model-name of '{model_name}'"
        )

    service_abbr = model_name.rsplit("-", 1)[-1]
    for cls in CLSRegistry.get_classes(FANTASY_SERVICE_DOMAIN):
        if cls.ABBR == service_abbr:
            break
    else:
        service_abbr = None
        _LOGGER.warning("Failed to parse service name from model-name '%s'", model_name)

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

    run_id, _ = train_and_log(
        experiment_name,
        (lambda: (model, model.name, metrics, model_artifacts, None)),
        experiment_description=experiment_description,
        run_name=final_run_name,
        run_description=run_description,
        tracker_settings=tracker_settings,
        run_tags=final_run_tags,
    )
    if set_active:
        _LOGGER.info(
            "Activating this model at run_id='%s' and deactivating all other models named '%s'",
            run_id,
            model_name,
        )
        deactivate_models(model_name=model_name, tracker_settings=tracker_settings)
        activate_model(run_id, tracker_settings=tracker_settings)
    return run_id


def cli_get_model(parser, args, tracker_settings: dict):
    if args.exp_name and args.run_id:
        parser.error("run-id and exp-name should not be used together")
    if not (args.exp_name or args.run_id or args.model_name or args.sport):
        print("No search criteria requested. Getting runs across all experiments")

    run_tags = {"sport": args.sport} if args.sport else {}

    retrieve(
        experiment_name=args.exp_name,
        run_id=args.run_id,
        model_name=args.model_name,
        active_only=not args.all,
        tracker_settings=tracker_settings,
        dest_path=args.dest,
        **run_tags,
    )


def cli_put_model(parser, args, tracker_settings: dict):
    archive_model(
        args.model_filepath,
        args.exp_name,
        experiment_description=args.exp_desc,
        run_name=args.run_name,
        run_description=args.run_desc,
        tracker_settings=tracker_settings,
        set_active=args.active,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Retrieve and archive models from model archive",
    )
    parser.add_argument(
        "--mlflow-uri",
        help=f"MLFlow tracking URI. default={os.environ.get('FANTASY_MLFLOW_TRACKING_URI')}",
        default=os.environ.get("FANTASY_MLFLOW_TRACKING_URI"),
    )
    parser.add_argument("--verbose", default=False, action="store_true")
    subparsers = parser.add_subparsers(help="Operation to perform")
    put_parser = subparsers.add_parser("put", help="Archive a model")
    put_parser.set_defaults(func=cli_put_model)
    put_parser.add_argument("model_filepath", metavar="model-filepath", help="path to .model file")
    put_parser.add_argument("--exp-name", help="experiment name")
    put_parser.add_argument("--exp-desc", help="experiment description")
    put_parser.add_argument("--run-name", help="run name")
    put_parser.add_argument("--run-desc", help="run description")
    put_parser.add_argument(
        "--active",
        help="set this model as active and deactivate conflicting models",
        action="store_true",
        default=False,
    )

    get_parser = subparsers.add_parser("get", help="Retrieve active models")
    get_parser.set_defaults(func=cli_get_model)
    get_parser.add_argument(
        "--exp-name",
        help="Limit to models in this experiment. "
        "If no other criteria is provided then look in the default experiment",
    )
    get_parser.add_argument(
        "--dest",
        help="Where models should be downloaded to. Default is to just list available models",
    )
    get_arg_group = get_parser.add_mutually_exclusive_group()
    get_arg_group.add_argument("--model-name", "--name", help="Model name")
    get_arg_group.add_argument("--run-id")
    get_arg_group.add_argument("--sport", help="Retrieve all models for this sport")
    get_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Default is to only retrieve active models, set this to retrieve all matching models",
    )

    args = parser.parse_args()
    if args.verbose:
        log.set_debug_log_level(only_fantasy=False)
        _LOGGER.info("Verbose mode enabled")
    if not hasattr(args, "func"):
        parser.error("No operation requested.")
    if not args.mlflow_uri:
        parser.error(
            "MLFlow URI must be set either on command line "
            "or in the envvar 'FANTASY_MLFLOW_TRACKING_URI'"
        )
    args.func(parser, args, tracker_settings={"mlf_tracking_uri": args.mlflow_uri})
