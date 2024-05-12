#! /venv/bin/python
import argparse
import glob
import json
import math
import os
import shlex
import sys
from typing import Literal, cast

import dask
import dateutil
import pandas as pd
from fantasy_py import UnexpectedValueError, dt_to_filename_str, log
from ledona import process_timer

from .pt_model import DEFAULT_DUMMY_REGRESSOR_KWARGS, AlgorithmType, TrainingConfiguration

_LOGGER = log.get_logger(__name__)


_CLITrainingParams = Literal[
    "training_mins",
    "training_iter_mins",
    "n_jobs",
    "early_stop",
    "max_epochs",
]
"""training parameters that are set on commandline and can override from command line during retrain"""


@process_timer
def _handle_train(args: argparse.Namespace):
    if not os.path.isdir(args.dest_dir):
        args.parser.error(f"Destination directory '{args.dest_dir}' does not exist.")
    args_dict = vars(args)

    if args.op == "train":
        tdf = TrainingConfiguration(filepath=args.cfg_file, algorithm=args.algorithm)
        original_model = None
        model_name = cast(str | None, args.model)
        if model_name is None:
            print(f"Following {len(tdf.model_names)} models are defined: {sorted(tdf.model_names)}")
            sys.exit(0)
        if model_name not in tdf.model_names:
            args.parser.error(
                f"Model '{model_name}' not defined. Try again with one "
                f"of the following: {tdf.model_names}"
            )
        assert model_name is not None
        cli_training_params = cast(
            dict[_CLITrainingParams, int | None],
            {k_: args_dict.get(k_) for k_ in _CLITrainingParams.__args__},
        )
    else:
        tdf, original_model = TrainingConfiguration.cfg_from_model(
            args.cfg_file, args.orig_cfg_file, algorithm=args.algorithm
        )
        model_name = original_model.name
        assert original_model.parameters is not None
        cli_training_params = cast(
            dict[_CLITrainingParams, int | None],
            {
                k_: args_dict.get(k_) or original_model.parameters.get(k_)
                for k_ in _CLITrainingParams.__args__
            },
        )

    if args.dask:
        _LOGGER.info("tpot dask enabled")
        dask.config.set(scheduler="processes", num_workers=math.floor(os.cpu_count() * 0.75))

    if tdf.algorithm.startswith("tpot"):
        modeler_init_kwargs = {
            "use_dask": args.dask,
            "n_jobs": cli_training_params["n_jobs"],
            "verbosity": 3,
        }
        if cli_training_params["training_mins"] is not None:
            modeler_init_kwargs["max_time_mins"] = cli_training_params["training_mins"]
        if cli_training_params["training_iter_mins"] is not None:
            modeler_init_kwargs["max_eval_time_mins"] = cli_training_params["training_iter_mins"]
        if cli_training_params["early_stop"] is not None:
            modeler_init_kwargs["early_stop"] = cli_training_params["early_stop"]
        if cli_training_params["max_epochs"] is not None:
            modeler_init_kwargs["generations"] = cli_training_params["max_epochs"]

    elif tdf.algorithm == "nn":
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # modeler_init_kwargs = {"device": device}
        modeler_init_kwargs = {
            key_[3:]: value_ for key_, value_ in vars(args).items() if key_.startswith("nn_")
        }
        if cli_training_params["early_stop"] is not None:
            modeler_init_kwargs["early_stop_epochs"] = cli_training_params["early_stop"]
        if cli_training_params["max_epochs"] is not None:
            modeler_init_kwargs["epochs_max"] = cli_training_params["max_epochs"]

    elif tdf.algorithm == "auto-xgb":
        modeler_init_kwargs = {
            "verbosity": 2,
            "n_jobs": cli_training_params["n_jobs"],
        }
        if cli_training_params["early_stop"] is not None:
            modeler_init_kwargs["early_stop_epochs"] = cli_training_params["early_stop"]
    elif tdf.algorithm == "dummy":
        modeler_init_kwargs = DEFAULT_DUMMY_REGRESSOR_KWARGS.copy()
    else:
        args.parse.error(f"Unknown algorithm '{tdf.algorithm}' requested")

    new_model = tdf._train_and_test(
        model_name,
        args.dest_dir,
        args.error_analysis_data,
        False if tdf.retrain else args.reuse,
        args.data_dir,
        args.info,
        args.dump_data,
        args.limited_data,
        args.dest_filename,
        **modeler_init_kwargs,
    )

    if not tdf.retrain:
        return

    assert original_model and new_model._input_cols and original_model._input_cols
    if new_model.target != original_model.target:
        raise UnexpectedValueError(
            f"new model target {new_model.target} does not match "
            f"original model target {original_model.target}!"
        )
    if (new_model_cols := set(new_model._input_cols)) != (
        orig_model_cols := set(original_model._input_cols)
    ):
        missing_cols = orig_model_cols - new_model_cols
        unexpected_cols = new_model_cols - orig_model_cols
        raise UnexpectedValueError(
            f"new model cols do not match original model cols! {missing_cols=} "
            f"{unexpected_cols=}"
        )


_MODEL_CATALOG_PATTERN = "model-catalog.{TIMESTAMP}.csv"


def _add_train_parser(sub_parsers):
    for op in ["train", "retrain"]:
        train_parser = sub_parsers.add_parser(op, help=f"{op} a model")
        train_parser.set_defaults(func=_handle_train, parser=train_parser, op=op)
        train_parser.add_argument(
            "cfg_file",
            help=f"{'.model' if op == 'train' else 'json'} file containing model configuration",
        )
        if op == "train":
            train_parser.add_argument(
                "model",
                nargs="?",
                help="Name of the model to train. If not set then model names will be listed. "
                "If the train file is a .model file this argument is ignored",
            )
            train_parser.add_argument(
                "--reuse",
                default=False,
                action="store_true",
                help="Reuse an model if it exists.",
            )
        else:
            train_parser.add_argument(
                "--orig_cfg_file",
                help="Original model definition json file. "
                "Only needed the .model file is missing training parameters",
            )
        train_parser.add_argument(
            "--info",
            default=False,
            action="store_true",
            help="Show final training parameters and stop (do not train)",
        )
        train_parser.add_argument(
            "--dump_data",
            metavar="filepath",
            help="Dump training data to a file. The file extension defines the format. "
            "Supported extensions: .csv, .pq",
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
            help="Algorithm for model selection/training. default={_DEFAULT_ALGORITHM}",
            choices=AlgorithmType.__args__,
        )
        train_parser.add_argument(
            "--n_jobs",
            "--tpot_jobs",
            help="Number of jobs/processors to use during training",
            type=int,
        )
        train_parser.add_argument(
            "--training_mins",
            "--mins",
            "--max_train_mins",
            "--time",
            "--max_time",
            "--max_time_mins",
            type=int,
            help="override the training time defined in the train_file",
        )
        train_parser.add_argument(
            "--training_iter_mins",
            "--max_iter_mins",
            "--iter_mins",
            "--iter_time",
            "--max_eval_time_mins",
            type=int,
            help="override the training iteration time defined in the train_file",
        )
        train_parser.add_argument(
            "--dest_dir", default=".", help="directory to write final model and artifact files to"
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
            help="limit the training data to this many sample",
        )
        train_parser.add_argument(
            "--early_stop",
            type=int,
            help="number of rounds/epochs of no improvement after which to stop training",
        )
        train_parser.add_argument(
            "--max_epochs",
            type=int,
            help="The maximum number of epochs to train a neural network model",
        )
        train_parser.add_argument(
            "--nn_checkpoint_dir", "--checkpoint_dir", help="The checkpoint directory for nn models"
        )
        train_parser.add_argument(
            "--nn_resume_checkpoint_filepath",
            "--resume_from",
            help="resume nn training from this checkpoint",
        )


def _model_catalog_func(args):
    """parser func that creates/updates the model catalog"""
    data = []

    glob_pattern = os.path.join(args.root, "**", "*.model")
    for model_filepath in glob.glob(glob_pattern, recursive=True):
        _LOGGER.info("parsing '%s'", model_filepath)
        with open(model_filepath, "r") as f_:
            model_data = json.load(f_)

        p_t = "player" if model_data["training_data_def"]["target"][1] == "P" else "team"
        data.append(
            {
                "name": model_data["name"],
                "sport": model_data["name"].split("-", 1)[0],
                "p/t": p_t,
                "dt": dateutil.parser.parse(model_data["dt_trained"]),
                "r2": model_data["meta_extra"]["performance"]["r2"],
                "mae": model_data["meta_extra"]["performance"]["mae"],
                "target": ":".join(model_data["training_data_def"]["target"]),
                "file": model_filepath,
            }
        )

    df = pd.DataFrame(data)

    filename = args.csv_filename or _MODEL_CATALOG_PATTERN.format(TIMESTAMP=dt_to_filename_str())
    df.to_csv(os.path.join(args.root, filename), index=False)

    if args.create_best_models_file:
        top_r2_df = df.groupby("name")["r2"].max().reset_index()
        best_models_df = df.merge(top_r2_df, on=["name", "r2"])[["name", "r2", "mae", "file"]]
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

    print()
    print(f"Catalog written to '{filename}'")
    if best_models_df is not None:
        print(f"Best models written to '{best_models_filename}'")


def _add_model_catalog_parser(sub_parsers):
    parser = sub_parsers.add_parser("catalog", help="Update model catalog")
    parser.set_defaults(func=_model_catalog_func, parser=parser)
    parser.add_argument(
        "--root",
        metavar="ROOT_DIRECTORY",
        help="The root directory to start the search for model files. Default=./",
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


def _model_load_actives_func(args):
    """parser func that loads active models"""
    raise NotImplementedError()


def _add_load_actives_parser(sub_parsers):
    parser = sub_parsers.add_parser("load", help="Load active models")
    parser.set_defaults(func=_model_load_actives_func, parser=parser)


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

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)
    if not hasattr(args, "func"):
        parser.print_help()
        parser.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
