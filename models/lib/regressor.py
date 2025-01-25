#! /venv/bin/python
import argparse
import glob
import json
import os
import re
import shlex
import sys
import traceback
from collections import defaultdict
from typing import Literal, cast

import pandas as pd
import tqdm
from dateutil import parser as du_parser
from fantasy_py import UnexpectedValueError, dt_to_filename_str, log
from fantasy_py.inference import PTPredictModel
from ledona import slack

from .pt_model import (
    DEFAULT_ALGORITHM,
    TRAINING_PARAM_DEFAULTS,
    AlgorithmType,
    ModelFileFoundMode,
    PerformanceOperation,
    TrainingConfiguration,
    performance_calc,
)

_LOGGER = log.get_logger(__name__)


_CLITrainingParams = Literal[
    "max_time_mins", "max_eval_time_mins", "n_jobs", "early_stop", "epochs_max", "population_size"
]
"""training parameters that are set on commandline and can be overriden
from command line during retrain"""


def _expand_models(
    tdf: TrainingConfiguration,
    model_request: str | list[str] | None,
):
    """figure out which models match the model requests"""
    if model_request is None:
        print(f"Following {len(tdf.model_names)} models are defined: {sorted(tdf.model_names)}")
        sys.exit(0)

    filters = [model_request] if isinstance(model_request, str) else model_request
    model_names: list[str] = []
    for model_filter in filters:
        if "*" not in model_filter:
            if model_filter not in tdf.model_names:
                raise UnexpectedValueError(
                    f"Model name '{model_filter}' not found. valid models are {tdf.model_names}"
                )
            model_names.append(model_filter)
            continue
        filter_re = model_filter.replace("*", ".*")
        matches = [model_name for model_name in tdf.model_names if re.match(filter_re, model_name)]
        if len(matches) == 0:
            raise UnexpectedValueError(
                f"Model request '{model_filter}' did not match to any model names. "
                f"valid models are {tdf.model_names}"
            )
        model_names += matches
    return model_names


def _algo_params(algo: AlgorithmType, use_dask, cli_training_params, args_dict: dict):
    if algo.startswith("tpot"):
        modeler_init_kwargs = {
            "use_dask": use_dask,
            "verbosity": 3,
        }
        rename = TRAINING_PARAM_DEFAULTS["tpot"][1] or {}
        for k_ in _CLITrainingParams.__args__:
            if k_ not in cli_training_params:
                continue
            key = rename.get(k_, k_)
            modeler_init_kwargs[key] = cli_training_params[k_]
        return modeler_init_kwargs

    if algo == "nn":
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # modeler_init_kwargs = {"device": device}
        modeler_init_kwargs = {
            key_[3:]: value_ for key_, value_ in args_dict.items() if key_.startswith("nn_")
        }
        assert (
            len(
                invalid_args := set(modeler_init_kwargs.keys())
                - set(TRAINING_PARAM_DEFAULTS["nn"][0].keys())
            )
            == 0
        ), f"nn_ cli args are not all valid. invalid_args={invalid_args}"
        rename = TRAINING_PARAM_DEFAULTS["nn"][1] or {}
        for k_ in _CLITrainingParams.__args__:
            if k_ not in cli_training_params:
                continue
            key = rename.get(k_, k_)
            modeler_init_kwargs[key] = cli_training_params[k_]
        return modeler_init_kwargs

    if algo == "xgboost":
        modeler_init_kwargs = {"verbosity": 2}
        return modeler_init_kwargs

    if algo == "dummy":
        return {}

    raise UnexpectedValueError(f"Unknown algorithm '{algo}' requested")


def _handle_train(args: argparse.Namespace):
    if not os.path.isdir(args.dest_dir):
        args.parser.error(f"Destination directory '{args.dest_dir}' does not exist.")

    args_dict = vars(args)

    model_names: list[str]
    if args.op == "train":
        tdf = TrainingConfiguration(filepath=args.cfg_file, algorithm=args.algorithm)
        if args.model is not None and args.models is not None:
            args.parser.error(
                f"Because model '{args.model}' was requested, --models cannot be used"
            )
        model_names = _expand_models(tdf, args.model or args.models)
        original_model = None
        cli_training_params = cast(
            dict[_CLITrainingParams, int | None],
            {k_: args_dict.get(k_) for k_ in _CLITrainingParams.__args__ if k_ in args_dict},
        )
    else:
        tdf, original_model = TrainingConfiguration.cfg_from_model(
            args.cfg_file, args.orig_cfg_file, algorithm=args.algorithm
        )
        model_names = [original_model.name]
        assert original_model.parameters is not None
        cli_training_params = {}
        for k_ in _CLITrainingParams.__args__:
            if k_ in args_dict:
                cli_training_params[k_] = args_dict[k_]
            elif k_ in original_model.parameters:
                cli_training_params[k_] = original_model.parameters[k_]

    _LOGGER.info("Training %i models. info-mode=%s: %s", len(model_names), args.info, model_names)

    if args.dask:
        _LOGGER.info("tpot dask enabled")

    modeler_init_kwargs = _algo_params(tdf.algorithm, args.dask, cli_training_params, args_dict)

    if args.slack and not args.info:
        slack.enable()
        slack.send_slack(f"Starting batch training for:{model_names}")
    else:
        slack.disable()

    progress: dict[str, list] = {"successes": [], "failures": []}
    for model_name in (
        p_bar := tqdm.tqdm(model_names, "models", disable=(len(model_names) == 1 or args.info))
    ):
        _LOGGER.info("Training %s", model_name)
        p_bar.set_postfix_str(
            f"{model_name} {log.GREEN}\u2714{log.COLOR_RESET}={len(progress['successes'])} "
            f"{log.RED}\u274C{log.COLOR_RESET}={len(progress['failures'])}"
        )
        try:
            new_model = tdf.train_and_test(
                model_name,
                args.dest_dir,
                args.error_analysis_data,
                args.exists_mode,
                args.data_dir,
                args.info,
                args.dump_data,
                args.limited_data,
                args.dest_filename,
                **modeler_init_kwargs,
            )
            progress["successes"].append(model_name)
        except RuntimeError as ex:
            # RuntimeError likely means a fail due to tpot
            _LOGGER.error("Failed to train '%s'", model_name, exc_info=ex)
            progress["failures"].append((model_name, ex))
            continue
        except BaseException as ex:
            msg = f"Unexpected error while training '{model_name}'. STOPPING"
            _LOGGER.critical(msg, exc_info=ex)
            slack.send_slack(
                msg + f": {ex}\n\n```" + traceback.format_exc(limit=None, chain=True) + "```"
            )
            raise

        if not tdf.retrain or args.info:
            continue
        assert (
            new_model is not None
            and original_model
            and new_model._input_cols
            and original_model._input_cols
        )
        if new_model.target != original_model.target:
            raise UnexpectedValueError(
                f"For '{model_name}' new model target {new_model.target} does not match "
                f"original model target {original_model.target}!"
            )
        if (new_model_cols := set(new_model._input_cols)) != (
            orig_model_cols := set(original_model._input_cols)
        ):
            missing_cols = orig_model_cols - new_model_cols
            unexpected_cols = new_model_cols - orig_model_cols
            _LOGGER.warning(
                "For '%s' new model cols do not match original model cols! missing_cols (n=%i) = "
                "%s unexpected_cols (n=%i) = %s",
                model_name,
                len(missing_cols),
                missing_cols,
                len(unexpected_cols),
                unexpected_cols,
            )
    if len(progress["failures"]) > 0:
        msg_prefix = (
            f"Only {len(model_names) - len(progress['failures'])} of {len(model_names)} "
            "models trained successfully. failed models="
        )
        l_msg = f"{msg_prefix}{progress['failures']}"
        final_slack_msg = f"{msg_prefix}{[failure[0] for failure in progress['failures']]}"
        _LOGGER.warning(l_msg)
    elif not args.info:
        final_slack_msg = (
            f"All models trained successfully: {model_names}"
            if len(model_names) > 1
            else f"{model_names[0]} trained successfully"
        )
        _LOGGER.success_info(final_slack_msg)
    if not args.info:
        slack.send_slack(final_slack_msg)


_MODEL_CATALOG_PATTERN = "model-catalog.{TIMESTAMP}.csv"


def _add_train_parser(sub_parsers):
    for op in ["train", "retrain"]:
        train_parser = sub_parsers.add_parser(op, help=f"{op} model(s)")
        train_parser.set_defaults(func=_handle_train, parser=train_parser, op=op)
        train_parser.add_argument(
            "cfg_file",
            help=f"{'.model' if op == 'train' else 'json'} file containing model configuration",
        )
        if op == "train":
            train_parser.add_argument(
                "model",
                nargs="?",
                help="Name of the model to train. Wildcard '*' is supported and will result "
                "in multiple models being trained.",
            )
            train_parser.add_argument(
                "--models", nargs="+", help="Models to train. Wildcard '*' is supported"
            )
        else:
            train_parser.add_argument(
                "--orig_cfg_file",
                help="Original model definition json file. "
                "Only needed the .model file is missing training parameters",
            )
        train_parser.add_argument(
            "--slack",
            help="send a slack notification on train start/end/fail",
            default=False,
            action="store_true",
        )
        train_parser.add_argument(
            "--info",
            default=False,
            action="store_true",
            help="Show final training parameters and stop (do not train)",
        )
        train_parser.add_argument(
            "--exists_mode",
            choices=ModelFileFoundMode.__args__,
            default="create-w-ts",
            help="What to do about existing model files",
        )
        train_parser.add_argument(
            "--dump_data",
            metavar="filepath",
            help="Dump training data to a file. The file extension defines the format. "
            "Supported extensions: .csv, .parquet",
        )
        train_parser.add_argument(
            "--error_analysis_data",
            help="Write error analysis data based on validation dataset to a file. "
            "Data consists of columns 'truth', 'prediction', 'error'",
            default=False,
            action="store_true",
        )
        train_parser.add_argument(
            "--algorithm",
            help=f"Algorithm for model selection/training. default='{DEFAULT_ALGORITHM}'",
            choices=AlgorithmType.__args__,
        )

        train_parser.add_argument(
            "--n_jobs",
            "--njobs",
            "--tpot_jobs",
            help="Number of jobs/processors to use during training",
            type=int,
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--max_time_mins",
            "--training_mins",
            "--mins",
            "--max_train_mins",
            "--time",
            "--max_time",
            type=int,
            default=argparse.SUPPRESS,
            help="override the training time defined in the train_file",
        )
        train_parser.add_argument(
            "--max_eval_time_mins",
            "--training_iter_mins",
            "--max_iter_mins",
            "--iter_mins",
            "--iter_time",
            type=int,
            default=argparse.SUPPRESS,
            help="override the training iteration time defined in the train_file",
        )
        train_parser.add_argument(
            "--population_size",
            type=int,
            default=argparse.SUPPRESS,
            help="Override population size for relevant models (e.g. tpot)",
        )
        train_parser.add_argument(
            "--dest_dir", default=".", help="Destination directory for model files"
        )
        train_parser.add_argument(
            "--dest_filename",
            help="The filename to write the model to. "
            "Model filenames will have the extension '.model'. "
            "If the requested filename does not have this extension it will be appended. "
            "Default is to use a filename based on the model name and a datetime stamp.",
        )
        train_parser.add_argument("--data_dir", help="The directory that data files are stored.")
        train_parser.add_argument("--dask", default=False, action="store_true")
        train_parser.add_argument(
            "--limited_data",
            "--data_limit",
            type=int,
            help="limit the training data to this many cases, "
            "validation data will not be limited",
        )
        train_parser.add_argument(
            "--early_stop",
            type=int,
            default=argparse.SUPPRESS,
            help="number of rounds/epochs/generations of no improvement "
            "after which to stop training",
        )
        train_parser.add_argument(
            "--epochs_max",
            "--max_epochs",
            "--generations",
            "--gens",
            type=int,
            default=argparse.SUPPRESS,
            help="The maximum number of epochs/generations to train a model",
        )

        # anything starting with nn_ will be for nn algo
        train_parser.add_argument(
            "--nn_batch_size",
            "--batch_size",
            type=int,
            help="Neural Network training batch size",
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--nn_hidden_size",
            "--hidden_size",
            type=int,
            help="Neural Network training batch size",
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--nn_learning_rate",
            "--learning_rate",
            "--lr",
            type=float,
            help="Neural Network training batch size",
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--nn_hidden_layers",
            "--hidden_layers",
            "--layers",
            type=int,
            help="nn layers",
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--nn_checkpoint_frequency",
            "--checkpoint_frequency",
            type=int,
            help="How often to save a periodic checkpoint (best models are always checkpointed)",
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--nn_checkpoint_dir",
            "--checkpoint_dir",
            help="The checkpoint directory for nn models",
            default=argparse.SUPPRESS,
        )
        train_parser.add_argument(
            "--nn_resume_checkpoint_filepath",
            "--resume_from",
            default=argparse.SUPPRESS,
            help="resume nn training from this checkpoint",
        )


def _model_catalog_func(args):
    """parser func that creates/updates the model catalog"""
    data = []
    glob_pattern = (
        os.path.join(args.root, "**", "*.model")
        if args.recursive
        else os.path.join(args.root, "*.model")
    )
    excluded_models: dict[str, list[str]] | None = defaultdict(list) if args.exclude_r else None

    for model_filepath in tqdm.tqdm(glob.glob(glob_pattern, recursive=True)):
        if excluded_models is not None:
            exclude = False
            for x_r in args.exclude_r:
                if re.match(x_r, model_filepath):
                    exclude = True
                    excluded_models[x_r].append(model_filepath)
                    _LOGGER.info(
                        "Skipping '%s' because it matches exclude pattern '%s'",
                        model_filepath,
                        x_r,
                    )
                    break
            if exclude:
                continue
        _LOGGER.info("parsing '%s'", model_filepath)
        with open(model_filepath, "r") as f_:
            model_data = json.load(f_)

        p_t = "player" if model_data["training_data_def"]["target"][1] == "P" else "team"
        if "r2_test" in model_data["meta_extra"]["performance"]:
            r2_train = model_data["meta_extra"]["performance"]["r2_train"]
            mae_train = model_data["meta_extra"]["performance"]["mae_train"]
            r2_test = model_data["meta_extra"]["performance"]["r2_test"]
            mae_test = model_data["meta_extra"]["performance"]["mae_test"]
            r2_val = model_data["meta_extra"]["performance"]["r2_val"]
            mae_val = model_data["meta_extra"]["performance"]["mae_val"]
        else:
            # TODO: this is for older models and eventually can be dropped
            r2_test = r2_train = mae_test = mae_train = None
            r2_val = model_data["meta_extra"]["performance"]["r2"]
            mae_val = model_data["meta_extra"]["performance"]["mae"]

        data.append(
            {
                "name": model_data["name"],
                "sport": model_data["name"].split("-", 1)[0],
                "p/t": p_t,
                "dt": du_parser.parse(model_data["dt_trained"]),
                "r2-val": r2_val,
                "mae-val": mae_val,
                "target": ":".join(model_data["training_data_def"]["target"]),
                "algo": model_data["parameters"].get("algorithm", DEFAULT_ALGORITHM),
                "filepath": model_filepath,
                "r2-test": r2_test,
                "mae-test": mae_test,
                "r2-train": r2_train,
                "mae-train": mae_train,
            }
        )

    df = pd.DataFrame(data).sort_values(by=["name", "dt"])

    filename = args.csv_filename or _MODEL_CATALOG_PATTERN.format(TIMESTAMP=dt_to_filename_str())
    df.to_csv(os.path.join(args.root, filename), index=False)

    if args.create_best_models_file:
        top_r2_df = df.groupby("name")["r2-val"].max().reset_index()
        best_models_df = (
            df.merge(top_r2_df, on=["name", "r2-val"])
            .sort_values(["name", "dt"], ascending=[True, False])
            .drop_duplicates("name", keep="first")
        )
        best_models_filename = f"best-models.{dt_to_filename_str()}.csv"
        best_models_df.to_csv(os.path.join(args.root, best_models_filename), index=False)
    else:
        best_models_filename = None
        best_models_df = None

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "expand_frame_repr",
        False,
    ):
        print(f"MODEL CATALOG (n={len(df)})")
        print(df.to_string(index=False))
        if best_models_df is not None:
            print()
            print(f"BEST MODELS IN CATALOG (n={len(best_models_df)})")
            print(best_models_df.to_string(index=False))

    _LOGGER.info("Catalog of n=%i models written to '%s'", len(df), filename)
    if best_models_df is not None:
        _LOGGER.info("Best models written to '%s'", best_models_filename)
    if excluded_models is not None:
        if len(excluded_models) == 0:
            _LOGGER.warning("Exclude patterns did not match any models!")
        else:
            _LOGGER.info(
                "Exclude patterns excluded %i models",
                sum(map(len, excluded_models.values())),
            )
            for x_r in args.exclude_r:
                if (num_excluded := len(excluded_models[x_r])) == 0:
                    _LOGGER.warning("  '%s' did not exclude any model files", x_r)
                    continue
                _LOGGER.info("  '%s' excluded %i model files", x_r, num_excluded)
                for model_path in excluded_models[x_r]:
                    _LOGGER.info("  '%s' excluded '%s'", x_r, model_path)


def _add_model_catalog_parser(sub_parsers):
    parser = sub_parsers.add_parser("catalog", help="Update model catalog")
    parser.set_defaults(func=_model_catalog_func, parser=parser)
    parser.add_argument(
        "--root",
        metavar="ROOT_DIRECTORY",
        help="The root directory to start the search for model files. "
        "Search is recursive. Default=./",
        default=".",
    )
    parser.add_argument(
        "--csv_filename",
        help="Specify the name that the CSV data will be saved to. "
        "File will be created in root directory. "
        f"Default filename will be '{_MODEL_CATALOG_PATTERN}'",
    )
    parser.add_argument(
        "--create_best_models_file",
        "--best",
        help="Create an file containing the best models for each model name based on r2. "
        "File will be created in root directory. ",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--exclude_r",
        nargs="+",
        help="Exclude model file paths that match these regular expressions",
    )
    parser.add_argument(
        "--not_recursive",
        "-nr",
        dest="recursive",
        default=True,
        action="store_false",
        help="Do not search recursively through directories for model files. "
        "Default is to search recursively",
    )


def _model_load_actives_func(args):
    """parser func that loads active models"""
    raise NotImplementedError()


def _add_load_actives_parser(sub_parsers):
    parser = sub_parsers.add_parser("load", help="Load active models")
    parser.set_defaults(func=_model_load_actives_func, parser=parser)


def _handle_performance(args):
    """performance recalculation"""
    if "*" in args.model_filepath:
        model_filepaths = glob.glob(args.model_filepath, recursive=True)
        if len(model_filepaths) == 0:
            _LOGGER.error("No model files matched '%s'", args.model_filepath)
            sys.exit(1)
    else:
        if not os.path.isfile(args.model_filepath):
            _LOGGER.error("No model file exists at '%s'", args.model_filepath)
            sys.exit(1)
        model_filepaths = [args.model_filepath]

    if args.max_missing_infer_cols is not None:
        if args.max_missing_infer_cols < 0 or args.max_missing_infer_cols > 1:
            args.parser.error("--max_missing_infer_cols must be between 0 and 1")
        PTPredictModel.MISSING_INFERENCE_COLS_THRESHOLD_FAIL = args.max_missing_infer_cols

    _LOGGER.info(
        "Executing performance op=%s on %i models matching '%s'",
        args.op,
        len(model_filepaths),
        args.model_filepath,
    )
    cfg = (
        TrainingConfiguration(args.train_cfg_filepath)
        if args.train_cfg_filepath is not None
        else None
    )
    performance_calc(args.op, model_filepaths, cfg, args.data_dir, args.skip_backups)


def _add_performance_parser(sub_parsers):
    parser = sub_parsers.add_parser("performance", help="Calculate/Update model performance")
    parser.set_defaults(func=_handle_performance, parser=parser)
    parser.add_argument(
        "model_filepath", help="path to the model's .model file, wildcard '*' is accepted"
    )
    parser.add_argument(
        "--train_cfg_filepath", "--cfg_filepath", "--cfg", help="The training config json file"
    )
    parser.add_argument("--data_dir", help="directory containing data files", default=".")
    parser.add_argument(
        "-op",
        choices=PerformanceOperation.__args__,
        default="test",
        help="calc=calculate and print new metrics; "
        "update=update model files with new metrics; "
        "repair=identify and update model files with incomplete metrics; "
        "test=identify model files with incomplete metrics",
    )
    parser.add_argument(
        "--max_missing_infer_cols",
        "--max_missing_features",
        type=float,
        help="Threshold for maximum percentage of missing inference features above which "
        "inference/prediction should fail. "
        f"Default={PTPredictModel.MISSING_INFERENCE_COLS_THRESHOLD_FAIL}",
    )
    parser.add_argument(
        "--skip_backups",
        default=False,
        action="store_true",
        help="When updating a model's performance, do not backup the model file",
    )


def main(cmd_line_str=None):
    log.set_default_log_level(only_fantasy=False)
    log.enable_progress()

    parser = argparse.ArgumentParser(
        description="Train and Test CLI for standard regression models"
    )
    subparsers = parser.add_subparsers()
    _add_train_parser(subparsers)
    _add_model_catalog_parser(subparsers)
    _add_load_actives_parser(subparsers)
    _add_performance_parser(subparsers)

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)
    if not hasattr(args, "func"):
        parser.print_help()
        parser.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
