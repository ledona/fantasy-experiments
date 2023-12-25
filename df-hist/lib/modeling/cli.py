import json
import logging
import os
import re
import shlex
from argparse import ArgumentParser
from itertools import product
from typing import Any

from fantasy_py import CONTEST_DOMAIN, CLSRegistry, ContestStyle
from fantasy_py.lineup.strategy import FiftyFifty, GeneralPrizePool
from tqdm import tqdm

from .. import log
from .automl import ExistingModelMode, Framework
from .eval_models import ModelTargetGroup, evaluate_models
from .results import show_eval_results

_LOGGER = logging.getLogger(__name__)
_DEFAULT_CFG_PATH = os.path.join(".", "model_cfg.json")
_DEFAULT_RESULTS_PATH = os.path.join(".", "eval_results")
_DEFAULT_MODEL_PATH = os.path.join(".", "models")


class _JSONWithCommentsDecoder(json.JSONDecoder):
    """
    json decoder that supports full line comments
    based on https://stackoverflow.com/a/72168909
    """

    def __init__(self, **kw):
        super().__init__(**kw)

    def decode(self, s: str) -> Any:
        s = "\n".join(l if not re.match(r"(//)|#.*", l.lstrip()) else "" for l in s.split("\n"))
        return super().decode(s)


def _multi_run(
    framework,
    model_params: dict,
    styles,
    sports,
    services,
    contest_types,
    models_to_test: set[ModelTargetGroup],
    model_folder,
    mode: ExistingModelMode,
):
    _LOGGER.info("starting multirun")
    models = {}
    eval_results = []
    if services is None:
        services = [None]
    progress_total = (
        len(sports)
        * len(styles)
        * len(services)
        * len(contest_types)
        * (len(models_to_test) if models_to_test else 6)
    )
    all_failed_models = []

    for sport, service, style, contest_type in (
        pbar := tqdm(
            product(sports, services, styles, contest_types), total=progress_total, desc="modeling"
        )
    ):
        (new_models, new_eval_results, failed_models) = evaluate_models(
            sport,
            style,
            contest_type,
            framework,
            model_params,
            pbar,
            model_folder=model_folder,
            models_to_test=models_to_test,
            service=service,
            mode=mode,
        )
        if failed_models:
            all_failed_models += failed_models
        if new_models is None:
            _LOGGER.warning(
                "No models generated for %s-%s-%s-%s",
                sport,
                service,
                style.name,
                contest_type.NAME,
            )
        else:
            models.update(new_models)
            eval_results += new_eval_results

    _LOGGER.info("finished multirun.")
    pbar.close()
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
        choices=[ContestStyle.CLASSIC, ContestStyle.SHOWDOWN],
        default=[ContestStyle.CLASSIC, ContestStyle.SHOWDOWN],
    )
    parser.add_argument(
        "--contest_types",
        "--types",
        nargs="+",
        choices=[FiftyFifty.NAME, GeneralPrizePool.NAME],
        default=[FiftyFifty.NAME, GeneralPrizePool.NAME],
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
        help=f"path where models will be written to. default={_DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--model_cfg_file",
        default=_DEFAULT_CFG_PATH,
        help=f"json file containing model configuration parameters. default={_DEFAULT_CFG_PATH}",
    )
    parser.add_argument(
        "--automl_framework",
        help="The type of automl algorithm to use. ",
        choices=Framework.__args__,
        default="tpot",
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

    if not os.path.isdir(args.results_path):
        parser.error(f"results path folder '{args.results_path}' does not exist")
    if not os.path.isfile(args.model_cfg_file):
        parser.error(f"model cfg file '{args.model_cfg_file}' does not exist")

    with open(args.model_cfg_file, "r") as f_:
        model_cfg = json.load(f_, cls=_JSONWithCommentsDecoder)
    if args.automl_framework not in model_cfg and args.automl_framework != "dummy":
        parser.error(
            f"automl type '{args.automl_framework}' not defined in model "
            f"cfg file '{args.model_cfg_file}'"
        )

    c_styles = [ContestStyle(style) for style in set(args.contest_styles)]
    c_types = [CLSRegistry.get_class(CONTEST_DOMAIN, type_) for type_ in set(args.contest_types)]

    print(f"{args=}")
    models, eval_results, failed_models = _multi_run(
        args.automl_framework,
        model_cfg.get(args.automl_framework, {}),
        c_styles,
        set(args.sports),
        set(args.services),
        c_types,
        set(args.model_types),
        args.model_path,
        args.existing_model_mode,
    )

    show_eval_results(eval_results, failed_models, args.results_path)


if __name__ == "__main__":
    log.setup()
    _process_cmd_line()
