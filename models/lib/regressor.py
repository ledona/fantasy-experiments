#! /venv/bin/python
import argparse
import glob
import json
import math
import os
import shlex
import sys
from pprint import pprint
from typing import Literal, TypedDict, cast

import dask
import dateutil
import pandas as pd
from fantasy_py import (
    FeatureType,
    JSONWithCommentsDecoder,
    PlayerOrTeam,
    UnexpectedValueError,
    dt_to_filename_str,
    log,
    typed_dict_validate,
)
from fantasy_py.inference import PTPredictModel
from ledona import process_timer

from .train_test import ArchitectureType, load_data, model_and_test

_LOGGER = log.get_logger(__name__)

_TPOT_TRAINING_PARAMS = Literal[
    "max_time_mins", "max_eval_time_mins", "n_jobs", "epochs_max", "early_stop"
]
"""names of parameters in a model definition file applicable to tpot"""
_TPOT_TRAINING_PARAMS_RENAME = {"epochs_max": "generations"}
"""remapping of model parameter names in definition file to tpot kwarg parameter names"""
_NN_TRAINING_PARAMS = Literal[
    "input_size",
    "hidden_size",
    "hidden_layers",
    "batch_size",
    "epochs_max",
    "early_stop_epochs",
    "learning_rate",
    "shuffle",
]
"""names of parameters in a model definition file applicable to NN models"""


class _Params(TypedDict):
    """definition of the final parameters used for model training/testing"""

    data_filename: str
    target: tuple[Literal["stat", "calc", "extra"], str] | str
    """target stat, either a tuple of (type, name) or string of 'type:name'"""
    validation_season: int
    recent_games: int
    training_seasons: list[int]

    # nullable/optional
    seed: None | int
    p_or_t: PlayerOrTeam | None
    include_pos: bool | None
    cols_to_drop: list[str] | None
    """columns/features to drop from training data. wildcards are accepted"""
    missing_data_threshold: float | None
    filtering_query: str | None
    """pandas compatible query string that will be run on the loaded data"""
    target_pos: list[str] | None
    training_pos: list[str] | None
    train_params: dict[_TPOT_TRAINING_PARAMS | _NN_TRAINING_PARAMS, int | str | float | bool] | None
    """params passed to the training algorithm (likely as kwargs)"""


class _TrainingDefinitionFile:
    # TODO: get rid of alt_def_file after model files are all up to date and contain all training data
    def __init__(self, file_path: str, alt_def_file: str | None):
        """
        file_path: path the the training config file. if the filename has a _model_ extension \
            then the configuration will be based off of the preexisting model
        alt_def_file: if file_path is a .model file, then this is used as an alternate cfg\
            file for any missing setting in the .model file.
        """
        if file_path.endswith(".model"):
            model = PTPredictModel.load(file_path)
            self._json = self._model_dict_from_model(model, alt_def_file)

        with open(file_path, "r") as f_:
            self._json = cast(dict, json.load(f_, cls=JSONWithCommentsDecoder))

        self._model_names_to_group_idx: dict[str, int] = {}
        for i, model_group in enumerate(self._json["model_groups"]):
            for model_name in model_group["models"]:
                self._model_names_to_group_idx[model_name] = i

    @staticmethod
    def _model_dict_from_model(model: PTPredictModel, data_filename: str) -> dict:
        """create a model training definition that reflects the model"""
        cfg_dict: dict = {
            "global_default": {},
            "model_groups": [
                {
                    "models": [
                        {
                            model.name: {
                                "data_filename": data_filename,
                                "training_seasons": 
                                # "missing_data_threshold": params.get("missing_data_threshold", 0),
                                # "filtering_query": params["filtering_query"],
                            }
                        }
                    ],
                }
            ],
        }
        raise NotImplementedError()
        return cfg_dict

    @property
    def model_names(self):
        return list(self._model_names_to_group_idx.keys())

    def get_params(self, model_name):
        """
        return a dict containing the training/evaluation parameters
        for the requested model
        """
        param_dict: dict = {
            param_key: None
            for param_key, value_type in _Params.__annotations__.items()
            if hasattr(value_type, "__args__") and type(None) in value_type.__args__
        }

        assert (
            param_dict["train_params"] is None
        ), "If the default for train params is not None then a dict update is needed"

        param_dict.update(self._json["global_default"].copy())
        if model_name not in self.model_names:
            raise UnexpectedValueError(f"'{model_name}' is not defined")

        model_group = self._json["model_groups"][self._model_names_to_group_idx[model_name]]
        param_dict.update(
            {k_: v_ for k_, v_ in model_group.items() if k_ not in ("train_params", "models")}
        )
        if model_group.get("train_params"):
            if not param_dict["train_params"]:
                param_dict["train_params"] = {}
            param_dict["train_params"].update(model_group["train_params"])

        param_dict.update(
            {k_: v_ for k_, v_ in model_group["models"][model_name].items() if k_ != "train_params"}
        )

        if model_train_params := model_group["models"][model_name].get("train_params"):
            if not param_dict["train_params"]:
                param_dict["train_params"] = {}
            param_dict["train_params"].update(model_train_params)
        param_dict["target"] = (
            tuple(param_dict["target"])
            if isinstance(param_dict["target"], list)
            else param_dict["target"]
        )
        param_dict["p_or_t"] = PlayerOrTeam(param_dict["p_or_t"]) if param_dict["p_or_t"] else None

        if validation_failure_reason := typed_dict_validate(_Params, param_dict):
            raise UnexpectedValueError(
                f"Model training parameter validation failure: {validation_failure_reason}"
            )
        return cast(_Params, param_dict)

    @staticmethod
    def _get_regressor_kwargs(arch: ArchitectureType, regressor_kwargs: dict, params: dict):
        # for any regressor kwarg not already set, fill in with model params
        if arch.startswith("tpot"):
            renamer = _TPOT_TRAINING_PARAMS_RENAME
            expected_param_names = set(_TPOT_TRAINING_PARAMS.__args__)
        elif arch == "nn":
            renamer: dict = {}
            expected_param_names = set(_NN_TRAINING_PARAMS.__args__)
        else:
            return regressor_kwargs

        new_kwargs = regressor_kwargs.copy()
        if not new_kwargs.get("random_state") and params["seed"]:
            new_kwargs["random_state"] = params["seed"]
        if params["train_params"]:
            if not set(params["train_params"].keys()) <= expected_param_names:
                _LOGGER.warning(
                    "Ignoring following parameters not used by '%s' models: %s",
                    arch,
                    set(params["train_params"].keys()) - expected_param_names,
                )
            for arg in expected_param_names:
                if new_kwargs.get(arg) or not params["train_params"].get(arg):
                    continue
                name = renamer.get(arg, arg)
                new_kwargs[name] = params["train_params"][arg]

        return new_kwargs

    @process_timer
    def _train_and_test(
        self,
        model_name: str,
        arch_type: ArchitectureType,
        dest_dir: str | None,
        error_data: bool,
        reuse_existing_models: bool,
        data_dir: str | None,
        info: bool,
        dump_data: str,
        limit: None | int,
        dest_filename: str | None,
        **regressor_kwargs,
    ):
        if error_data:
            raise NotImplementedError()
        params = self.get_params(model_name)

        data_filepath = params["data_filename"]
        if data_dir is not None:
            data_filepath = os.path.join(data_dir, data_filepath)

        _LOGGER.info(
            "loading data from '%s'%s", data_filepath, f" with limit {limit}" if limit else ""
        )

        _, tt_data, one_hot_stats = load_data(
            data_filepath,
            params["target"],
            params["validation_season"],
            params["seed"],
            include_position=params["include_pos"],
            col_drop_filters=params["cols_to_drop"],
            missing_data_threshold=params.get("missing_data_threshold", 0),
            filtering_query=params["filtering_query"],
            limit=limit,
        )

        _LOGGER.info(
            "data load of '%s' complete. one_hot_stats=%s", params["data_filename"], one_hot_stats
        )
        if dump_data:
            _LOGGER.info("Dumping training data to '%s'", dump_data)
            df = pd.concat(tt_data[0:2], axis=1)
            if dump_data.endswith(".csv"):
                df.to_csv(dump_data)
            elif dump_data.endswith(".pq"):
                df.to_parquet(dump_data)
            else:
                raise UnexpectedValueError(f"Unknown data dump format: {dump_data}")

        final_regressor_kwargs = self._get_regressor_kwargs(arch_type, regressor_kwargs, params)
        print("\nTraining will proceed with the following parameters:")
        pprint(params)
        print()
        _LOGGER.info(
            "Fitting model '%s' with arch=%s using final_regressor_kwargs=%s",
            model_name,
            arch_type,
            final_regressor_kwargs,
        )

        if info:
            print(f"\nModel parameters for {model_name}")
            pprint(params)
            print(f"Data features (n={len(tt_data[0].columns)}): {sorted(tt_data[0].columns)}")
            sys.exit(0)

        target = (
            params["target"]
            if isinstance(params["target"], (tuple, list))
            else params["target"].split(":")
        )
        if len(target) != 2 or target[0] not in FeatureType.__args__:
            raise UnexpectedValueError(f"Invalid model target defined in cfg file: {target}")

        misc_params = {
            "missing_data_threshold": params.get("missing_data_threshold", 0),"filtering_query": params["filtering_query"]
        }

        model = model_and_test(
            model_name,
            params["validation_season"],
            tt_data,
            cast(tuple[FeatureType, str], target),
            arch_type,
            params["p_or_t"],
            params["recent_games"],
            params["training_seasons"],
            final_regressor_kwargs,
            params["target_pos"],
            params["training_pos"] or params["target_pos"],
            dest_dir,
            reuse_existing_models,
            model_dest_filename=dest_filename,
            misc_params=misc_params,
        )

        return model


_DEFAULT_ARCHITECTURE: ArchitectureType = "tpot"
_DUMMY_REGRESSOR_KWARGS = {"strategy": "median"}


def _handle_train(args):
    if args.train_file.endswith(".model") and args.reuse is True:
        args.parser.error("--reuse cannot be used with a .model file")

    tdf = _TrainingDefinitionFile(args.train_file, args.original_cfg_file)

    if not os.path.isdir(args.dest_dir):
        _LOGGER.critical("Destination directory '%s' does not exist.", args.dest_dir)
        sys.exit(0)

    if not args.model:
        print(f"Following {len(tdf.model_names)} models are defined: {sorted(tdf.model_names)}")
        sys.exit(0)

    if args.model not in tdf.model_names:
        args.parser.error(
            f"Model '{args.model}' not defined. Try again with one "
            f"of the following: {tdf.model_names}"
        )

    if args.dask:
        _LOGGER.info("tpot dask enabled")
        dask.config.set(scheduler="processes", num_workers=math.floor(os.cpu_count() * 0.75))

    if args.arch.startswith("tpot"):
        modeler_init_kwargs = {"use_dask": args.dask, "n_jobs": args.n_jobs, "verbosity": 3}
        if args.training_mins is not None:
            modeler_init_kwargs["max_time_mins"] = args.training_mins
        if args.training_iter_mins is not None:
            modeler_init_kwargs["max_eval_time_mins"] = args.training_iter_mins
        if args.early_stop is not None:
            modeler_init_kwargs["early_stop"] = args.early_stop
        if args.max_epochs is not None:
            modeler_init_kwargs["generations"] = args.max_epochs

    elif args.arch == "nn":
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # modeler_init_kwargs = {"device": device}
        modeler_init_kwargs = {
            key_[3:]: value_ for key_, value_ in vars(args).items() if key_.startswith("nn_")
        }
        if args.early_stop is not None:
            modeler_init_kwargs["early_stop_epochs"] = args.early_stop
        if args.max_epochs is not None:
            modeler_init_kwargs["epochs_max"] = args.max_epochs

    elif args.arch == "automl-xgb":
        modeler_init_kwargs = {
            "verbosity": 2,
            "n_jobs": args.n_jobs,
        }
        if args.early_stop is not None:
            modeler_init_kwargs["early_stop_epochs"] = args.early_stop
    elif args.arch == "dummy":
        modeler_init_kwargs = _DUMMY_REGRESSOR_KWARGS.copy()
    else:
        args.parse.error(f"Unknown architecture '{args.arch}' requested")

    tdf._train_and_test(
        args.model,
        cast(ArchitectureType, args.arch),
        args.dest_dir,
        args.error_analysis_data,
        args.reuse,
        args.data_dir,
        args.info,
        args.dump_data,
        args.limited_data,
        args.dest_filename,
        **modeler_init_kwargs,
    )


_MODEL_CATALOG_PATTERN = "model-catalog.{TIMESTAMP}.csv"


def _add_train_parser(sub_parsers):
    train_parser = sub_parsers.add_parser("train", help="Train a model")
    train_parser.set_defaults(func=_handle_train, parser=train_parser)
    train_parser.add_argument(
        "train_file",
        help="Json file containing training parameters or a .model file for an existing model",
    )
    train_parser.add_argument(
        "model",
        nargs="?",
        help="Name of the model to train. If not set then model names will be listed. "
        "If the train file is a .model file this argument is ignored",
    )
    train_parser.add_argument("--original_cfg_file", help="if the train_file is a .model file then anything that cannot be inferred from the .model file will be retrieved from here")
    train_parser.add_argument("--info", default=False, action="store_true")
    train_parser.add_argument(
        "--dump_data",
        metavar="filepath",
        help="Dump training data to a file. The file extension defines the format. "
        "Supported extensions: .csv, .pq",
    )
    train_parser.add_argument(
        "--reuse",
        default=False,
        action="store_true",
        help="Only useable if the train file is a model config json (not a .model file)",
    )
    train_parser.add_argument(
        "--error_analysis_data",
        help="Write error analysis data based on validation dataset to a file. "
        "Data consists of columns 'truth', 'prediction', 'error'",
        default=False,
        action="store_true",
    )
    train_parser.add_argument(
        "--arch",
        default=_DEFAULT_ARCHITECTURE,
        choices=ArchitectureType.__args__,
        help="model architecture",
    )
    train_parser.add_argument(
        "--n_jobs", "--tpot_jobs", help="Number of jobs/processors to use during training", type=int
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
