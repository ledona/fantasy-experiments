"""functions required for exporting and training deep models"""

import argparse
import os
import shlex
from typing import cast

from fantasy_py import FANTASY_SERVICE_DOMAIN, CacheMode, CLSRegistry, ContestStyle, db, log
from fantasy_py.lineup import FantasyService

from .deep import ExistingFilesMode, deep_data_export, deep_train

_DEFAULT_SAMPLE_COUNT = 10
_DEFAULT_PARENT_DATASET_PATH = "/fantasy-isync/fantasy-modeling/deep_lineup"
_DEFAULT_GAMES_PER_SLATE = 4, 6


def _data_export_parser_func(args: argparse.Namespace, parser: argparse.ArgumentParser):
    db_obj = db.get_db_obj(args.db_file)
    service_class = cast(
        FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, args.service)
    )

    iters: list[tuple[list[int], int, str]] = []
    if not args.skip_training:
        iters.append((args.seasons, args.samples, "-train"))
    if args.validation is not None:
        try:
            season, cases = args.validation
            cases = int(args.samples * float(cases) if "." in cases else cases)
            iters.append(([int(season)], cases, "-test"))
        except ValueError as ex:
            parser.error(f"Failed to parse --validation argument: {ex}")

    for seasons, samples, suffix in iters:
        deep_data_export(
            db_obj,
            args.name + suffix,
            args.dest_dir,
            seasons,
            args.games_per_slate_range,
            samples,
            args.existing_files_mode,
            service_class,
            args.style,
            args.seed,
            args.cache_dir,
            args.cache_mode if args.cache_dir else "disable",
        )


def _train_parser_func(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.model_filepath:
        # write to [dataset_dir]/[default_model_filename]
        target_dir = os.path.join("..", args.dataset_dir)
        model_filename = None
    elif os.path.isdir(args.model_filepath):
        # write to [model_filepath]/[default_model_filename]
        target_dir = args.model_filepath
        model_filename = None
    else:
        # write to [model_filepath]
        target_dir = os.path.dirname(args.model_filepath)
        model_filename = os.path.basename(args.model_filepath)
    if not os.path.isdir(target_dir):
        parser.error(f"Infered model target directory '{target_dir}' is not a valid directory!")

    deep_train(
        args.dataset_dir,
        args.epochs,
        args.batch_size,
        target_dir,
        model_filename,
        hidden_size=args.hidden_size,
        continue_from_checkpoint_filepath=args.checkpoint_filepath,
        checkpoint_epoch_interval=args.checkpoint_frequency,
        dataset_limit=args.dataset_limit,
        early_stopping_patience=args.early_stopping_patience,
    )


def main(cmd_line_str=None):
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

    sub_parsers = parser.add_subparsers(
        title="operation", help="The deep learning lineup operation to execute"
    )

    data_parser = sub_parsers.add_parser(
        "data", help="Create training data for deep learning lineup models"
    )
    data_parser.set_defaults(func=_data_export_parser_func)
    data_parser.add_argument("--cache_dir", default=None, help="Folder to cache to")
    data_parser.add_argument("--cache_mode", choices=CacheMode.__args__)
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
        nargs="+",
        type=int,
        help="Default is to infer slates from any available season",
    )
    data_parser.add_argument(
        "--games_per_slate_range",
        help=f"Number of games per slate. Default is {_DEFAULT_GAMES_PER_SLATE}",
        nargs=2,
        type=int,
        default=_DEFAULT_GAMES_PER_SLATE,
    )
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
    data_parser.add_argument("db_file")

    service_names = CLSRegistry.get_names(FANTASY_SERVICE_DOMAIN)
    service_abbrs = {
        CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, name).ABBR.lower(): name
        for name in service_names
    }
    data_parser.add_argument("service", choices=sorted(service_names + list(service_abbrs.keys())))

    train_parser = sub_parsers.add_parser("train", help="Train a deep model")
    train_parser.set_defaults(func=_train_parser_func)
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
        "If this is a directory then the model will be written to '[model_filepath]/{_DEFAULT_MODEL_FILENAME_FORMAT}'. "
        "If this is a filename without a path then the model will be written to '[dataset_dir]/[model_filepath]'. "
        "default='[dataset_dir]/../{_DEFAULT_MODEL_FILENAME_FORMAT}'",
    )
    train_parser.add_argument("--model_dir", help="The directory to")

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    if args.progress:
        log.enable_progress()

    if args.verbose:
        log.set_debug_log_level()

    args.func(args, parser)


if __name__ == "__main__":
    main()
