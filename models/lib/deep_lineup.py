"""functions required for exporting and training deep models"""

import argparse
import os
import shlex
from typing import cast

from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CacheSettings,
    CLSRegistry,
    ContestStyle,
    db,
    log,
)
from fantasy_py.lineup import FantasyService
from ledona import slack

from .deep import GEN_LINEUP_HELPER_FUNC_LABEL, ExistingFilesMode, deep_data_export, deep_train

_DEFAULT_SAMPLE_COUNT = 10
_DEFAULT_PARENT_DATASET_PATH = "/fantasy-isync/fantasy-modeling/deep_lineup"


def _data_export_parser_func(args: argparse.Namespace, parser: argparse.ArgumentParser):
    db_obj = db.get_db_obj(args.db_file)
    service_class = cast(
        FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, args.service)
    )

    iters: list[tuple[tuple[int, int] | int, int, str]] = []
    if not args.skip_training:
        iters.append((args.seasons, args.samples, "-train"))
    if args.validation is not None:
        try:
            season, cases = args.validation
            cases = int(args.samples * float(cases) if "." in cases else cases)
            iters.append((int(season), cases, "-test"))
        except ValueError as ex:
            parser.error(f"Failed to parse --validation argument: {ex}")

    slack.send_slack(f"Starting deep-lineup data export name={args.name}")

    try:
        for seasons, samples, suffix in iters:
            deep_data_export(
                db_obj,
                args.name + suffix,
                args.dest_dir,
                seasons,
                (args.games_per_slate_min, args.games_per_slate_max),
                samples,
                args.existing_files_mode,
                service_class,
                args.style,
                args.seed,
                args.batch_size,
            )
    except Exception as ex:
        slack.send_slack(f"Unhandled failure during deep-lineup data export name={args.name}: {ex}")
        raise
    slack.send_slack(f"Successful completion for deep-lineup data export name={args.name}")


def _train_parser_func(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.model_filepath:
        # write to [dataset_dir]/[default_model_filename]
        target_dir = os.path.join("..", args.dataset_dir)
    elif os.path.isdir(args.model_filepath):
        # write to [model_filepath]/[default_model_filename]
        target_dir = args.model_filepath
    elif not os.path.exists(args.model_filepath):
        create_dir = input(
            f"Model target directory '{args.model_filepath}' does not exist. Try and create? [y/N] "
        )
        if create_dir.upper() != "Y":
            parser.error(f"Model target directory '{args.model_filepath}' does not exist.")
        os.mkdir(args.model_filepath)
        print(f"Created model target directory '{args.model_filepath}'")
        target_dir = args.model_filepath
    else:
        parser.error(
            f"Unable to write to model target directory '{args.model_filepath}'. "
            "Perhaps this path points to a file?"
        )

    if not os.path.isdir(target_dir):
        parser.error(f"Infered model target directory '{target_dir}' is not a valid directory!")

    slack.send_slack(f"Starting deep-lineup training with dataset '{args.dataset_dir}'")

    try:
        deep_train(
            args.dataset_dir,
            args.epochs,
            args.batch_size,
            target_dir,
            hidden_size=args.hidden_size,
            continue_from_checkpoint_filepath=args.checkpoint_filepath,
            checkpoint_epoch_interval=args.checkpoint_frequency,
            dataset_limit=args.dataset_limit,
            early_stopping_patience=args.early_stopping_patience,
            overwrite=args.overwrite,
        )
    except Exception as ex:
        slack.send_slack(
            f"Unhandled failure during deep-lineup training with dataset '{args.dataset_dir}': {ex}"
        )
        raise
    slack.send_slack(
        f"Successful completion for deep-lineup training with dataset '{args.dataset_dir}'"
    )


def _add_data_parser_args(data_parser: argparse.ArgumentParser):
    data_parser.set_defaults(func=_data_export_parser_func, op="data")

    CacheSettings.update({"timeout": 60 * 60 * 24 * 7}, func_label=GEN_LINEUP_HELPER_FUNC_LABEL)
    CacheSettings.add_parser_args(data_parser)

    data_parser.add_argument(
        "--samples",
        "--cases",
        "--n",
        type=int,
        help=f"How many samples to infer. default={_DEFAULT_SAMPLE_COUNT}",
        default=_DEFAULT_SAMPLE_COUNT,
    )

    data_parser.add_argument("--style", default=ContestStyle.CLASSIC)
    data_parser.add_argument(
        "--seasons",
        nargs=2,
        type=int,
        metavar=("first-season", "last-season"),
        help="Default is to infer slates from all available seasons",
    )
    data_parser.add_argument(
        "--games_per_slate_min",
        "--min_games",
        help="minimum number of games per slate. Default is infered from the most recent season.",
        type=int,
    )
    data_parser.add_argument(
        "--games_per_slate_max",
        "--max_games",
        help="maxiumum number of games per slate. Default is infered from the most recent season.",
        type=int,
    )
    data_parser.add_argument("--batch_size", default=10)
    data_parser.add_argument(
        "--existing_files_mode",
        "--file_mode",
        choices=ExistingFilesMode.__args__,
        default="append",
        help="Default is append, new files will overwrite or add to existing "
        "files in the dataset directory",
    )
    data_parser.add_argument("--name", help="Name of the dataset. Default is a datetime stamp")
    data_parser.add_argument(
        "--validation",
        nargs=2,
        metavar=("validation-season", "validation-cases"),
        help="Create an additional validation set of this size from this season",
    )
    data_parser.add_argument(
        "--skip_training", action="store_true", help="Skip training data creation", default=False
    )
    data_parser.add_argument("--disable_dask", default=False, action="store_true")

    data_parser.add_argument("db_file")

    service_names = CLSRegistry.get_names(FANTASY_SERVICE_DOMAIN)
    service_abbrs = {
        CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, name).ABBR.lower(): name
        for name in service_names
    }
    data_parser.add_argument("service", choices=sorted(service_names + list(service_abbrs.keys())))


def _add_train_parser_args(train_parser: argparse.ArgumentParser):
    train_parser.set_defaults(func=_train_parser_func, op="train")
    train_parser.add_argument(
        "dataset_dir",
        help="Path to the training/testing dataset. "
        "There should be two folders one for training that ends in '-train' and another for "
        "test that ends in '-test'",
    )
    train_parser.add_argument(
        "--dataset_limit", "--limit", type=int, help="Limit the data set to this many samples"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=32, help="The number of samples/slates per batch"
    )
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Number of epochs without improvement to wait before stopping training early. "
        "Default is no early stopping",
    )
    train_parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers")
    train_parser.add_argument("--checkpoint_filepath", help="Path to checkpoint to continue from")
    train_parser.add_argument(
        "--checkpoint_frequency",
        default=10,
        type=int,
        help="Frequency of checkpoints (in epochs). Checkpoints are also made for the last "
        "training epoch and whenever a new best score model is found",
    )
    train_parser.add_argument(
        "--model_filepath",
        help="Filename to write model to. Default is to write to. "
        "If this is a directory then the model will be written to "
        "'[model_filepath]/{_DEFAULT_MODEL_FILENAME_FORMAT}'. "
        "If this is a filename without a path then the model will "
        "be written to '[dataset_dir]/[model_filepath]'. "
        "default='[dataset_dir]/../{_DEFAULT_MODEL_FILENAME_FORMAT}'",
    )
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing model file if one already exists",
    )


def _process_cmd_line(cmd_line_str=None):
    log.set_default_log_level(only_fantasy=False)

    parser = argparse.ArgumentParser(
        description="Functions to export data for, train and test deep "
        "learning lineup generator models"
    )

    parser.add_argument("--seed", default=0)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--dest_dir",
        help="Directory under which dataset directories will be written. "
        f"Default is '{_DEFAULT_PARENT_DATASET_PATH}'",
        default=_DEFAULT_PARENT_DATASET_PATH,
    )
    parser.add_argument("--no_progress", dest="progress", default=True, action="store_false")
    parser.add_argument(
        "--slack",
        help="send a slack notification on data generation start/end/fail",
        default=False,
        action="store_true",
    )

    sub_parsers = parser.add_subparsers(
        title="operation", help="The deep learning lineup operation to execute"
    )

    data_parser = sub_parsers.add_parser(
        "data", help="Create training data for deep learning lineup models"
    )
    _add_data_parser_args(data_parser)

    train_parser = sub_parsers.add_parser("train", help="Train a deep model")
    _add_train_parser_args(train_parser)

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    if not hasattr(args, "func"):
        parser.print_help()
        parser.exit(1)

    if args.progress:
        log.enable_progress()

    if args.verbose:
        log.set_debug_log_level()

    if args.slack:
        slack.enable()
    else:
        slack.disable()

    return parser, args


if __name__ == "__main__":
    parser_, args_ = _process_cmd_line()
    if args_.op == "data" and not args_.disable_dask:
        print("Starting distributed dask for data export")
        from dask.distributed import Client

        client = Client(processes=True)
        CacheSettings.register_dask(client)

    args_.func(args_, parser_)
