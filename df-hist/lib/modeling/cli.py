import json
import os
import shlex
from argparse import ArgumentParser
from itertools import product

from fantasy_py import CONTEST_DOMAIN, CLSRegistry, DFSContestStyle, JSONWithCommentsDecoder, log
from fantasy_py.betting import FiftyFifty, GeneralPrizePool
from tqdm import tqdm

from .eval_models import ModelTargetGroup, evaluate_models
from .model import ExistingModelMode, Framework
from .results import show_eval_results

_LOGGER = log.get_logger(__name__)
_DEFAULT_CFG_PATH = os.path.join(".", "model_cfg.json")
_DEFAULT_RESULTS_PATH = os.path.join(".", "eval_results")
_DEFAULT_MODEL_PATH = os.path.join(".", "models")


def _multi_run(
    frameworks: list[Framework],
    model_cfg_filepath,
    styles,
    sports,
    services,
    contest_types,
    models_to_test: set[ModelTargetGroup],
    model_folder,
    mode: ExistingModelMode,
    eval_results_path: str,
    data_dir: str | None,
):
    """
    generate multiple models for combinations of requested styles,
    sports, services and contest_types
    """
    _LOGGER.info("starting multirun for %s-%s-%s-%s", sports, services, styles, contest_types)

    with open(model_cfg_filepath, "r") as f_:
        model_cfgs = json.load(f_, cls=JSONWithCommentsDecoder)

    models = {}
    eval_results = []
    if services is None:
        services = [None]
    progress_total = (
        len(sports)
        * len(styles)
        * len(services)
        * len(contest_types)
        * len(frameworks)
        * (len(models_to_test) if models_to_test else 6)
    )
    all_failed_models = []

    pbar = tqdm(
        product(sorted(sports), sorted(services), styles, contest_types, sorted(frameworks)),
        total=progress_total,
        desc="modeling",
    )
    for sport, service, style, contest_type, framework in pbar:
        if framework not in model_cfgs:
            raise ValueError(
                f"framework '{framework}' not defined in model " f"cfg file '{model_cfg_filepath}'"
            )

        framework_params = model_cfgs[framework] or {}

        (new_models, new_eval_results, failed_models) = evaluate_models(
            sport,
            style,
            contest_type,
            framework,
            framework_params,
            pbar,
            model_folder=model_folder,
            eval_results_path=eval_results_path,
            models_to_test=models_to_test,
            service=service,
            mode=mode,
            data_folder=data_dir,
        )
        if failed_models:
            all_failed_models += failed_models
        if new_models is None:
            _LOGGER.warning(
                "No models generated for %s-%s-%s-%s-%s",
                framework,
                sport,
                service,
                style.name,
                contest_type.NAME,
            )
            continue
        models.update(new_models)
        eval_results += new_eval_results

    _LOGGER.info(
        "finished multirun of %s-%s-%s-%s-%s", frameworks, sports, services, styles, contest_types
    )
    return models, eval_results, all_failed_models


def _process_cmd_line(cmd_line_str=None):
    parser = ArgumentParser(description="Train and evaluate df min/max score models")
    parser.add_argument(
        "--services",
        help="default='draftkings'",
        nargs="+",
        choices=["draftkings", "fanduel", "yahoo"],
        default=["draftkings"],
    )
    parser.add_argument(
        "--contest_styles",
        "--styles",
        nargs="+",
        choices=[DFSContestStyle.CLASSIC, DFSContestStyle.SHOWDOWN],
        default=[DFSContestStyle.CLASSIC, DFSContestStyle.SHOWDOWN],
    )
    parser.add_argument(
        "--contest_types",
        "--types",
        nargs="+",
        choices=[FiftyFifty.TYPE_NAME, GeneralPrizePool.TYPE_NAME],
        default=[FiftyFifty.TYPE_NAME, GeneralPrizePool.TYPE_NAME],
    )
    parser.add_argument(
        "--model_types",
        nargs="+",
        choices=ModelTargetGroup.__args__,
        default=ModelTargetGroup.__args__,
    )
    parser.add_argument(
        "--results_path",
        default=_DEFAULT_RESULTS_PATH,
        help=f"path where evaluation results will be written to. default={_DEFAULT_RESULTS_PATH}",
    )
    parser.add_argument(
        "--model_path",
        default=_DEFAULT_MODEL_PATH,
        help=f"path where models will be written to. default='{_DEFAULT_MODEL_PATH}'",
    )
    parser.add_argument("--data_dir", "--data_path", help="default is ./data", default="data")
    parser.add_argument(
        "--model_cfg_file",
        default=_DEFAULT_CFG_PATH,
        help=f"json file containing model configuration parameters. default={_DEFAULT_CFG_PATH}",
    )
    parser.add_argument(
        "--frameworks",
        help="The type of ml framework/algorithm to use",
        choices=Framework.__args__,
        nargs="+",
        default=["tpot"],
    )
    parser.add_argument(
        "--existing_model_mode",
        "--mode",
        choices=ExistingModelMode.__args__,
        default="fail",
        help="What to do if a model already exists. reuse=evaluate the model; "
        "overwrite=retrain the model; fail=fail with file already exists",
    )
    parser.add_argument("sports", nargs="+", choices=["nhl", "nfl", "mlb", "nba", "lol"])

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    if args.data_dir is not None and not os.path.isdir(args.data_dir):
        parser.error(f"data directory '{args.data_dir}' is not a folder")
    if not os.path.isdir(args.results_path):
        parser.error(f"results path folder '{args.results_path}' does not exist")
    if not os.path.isfile(args.model_cfg_file):
        parser.error(f"model cfg file '{args.model_cfg_file}' does not exist")
    if not os.path.isdir(args.model_path):
        parser.error(f"model destination path '{args.model_path}' does not exist")

    c_styles = [DFSContestStyle(style) for style in set(args.contest_styles)]
    c_types = [CLSRegistry.get_class(CONTEST_DOMAIN, type_) for type_ in set(args.contest_types)]

    print(f"{args=}")
    eval_results, failed_models = _multi_run(
        args.frameworks,
        args.model_cfg_file,
        c_styles,
        set(args.sports),
        set(args.services),
        c_types,
        set(args.model_types),
        args.model_path,
        args.existing_model_mode,
        args.results_path,
        args.data_dir,
    )[1:]

    show_eval_results(eval_results, failed_models, args.results_path)


if __name__ == "__main__":
    log.configure_logging(progress=True)
    log.set_log_level(log.LIMITED_INFO, only_fantasy=False)
    _process_cmd_line()
