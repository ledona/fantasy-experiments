#! /venv/bin/python
import argparse
import json
import os
import pathlib
import shlex
import sys
from pprint import pprint
from typing import Literal, TypedDict, cast

from fantasy_py import JSONWithCommentsDecoder, PlayerOrTeam, typed_dict_validate
from ledona import process_timer

from .train_test import AutomlType, OverwriteMode, load_data, model_and_test

_EXPECTED_TRAINING_PARAMS = Literal["max_time_mins", "max_eval_time_mins", "n_jobs"]


class _Params(TypedDict):
    """definition of the final parameters used for model training/testing"""

    data_filename: str
    target: tuple[Literal["stat", "calc", "extra"], str]
    validation_season: int
    recent_games: int
    training_seasons: list[int]

    # nullable/optional
    seed: None | int
    p_or_t: PlayerOrTeam | None
    include_pos: bool | None
    cols_to_drop: list[str] | None
    """columns to drop from inout data. wildcards are accepted"""
    missing_data_threshold: float | None
    filtering_query: str | None
    """pandas compatible query string that will be run on the loaded data"""
    target_pos: list[str] | None
    training_pos: list[str] | None
    train_params: dict[_EXPECTED_TRAINING_PARAMS, int | str | float | bool] | None
    """params passed to the training algorithm (likely as kwargs)"""


class TrainingDefinitionFile:
    def __init__(self, file_path: str):
        self._file_path = file_path
        with open(file_path, "r") as f_:
            self._json = cast(dict, json.load(f_, cls=JSONWithCommentsDecoder))

        self._model_names_to_group_idx: dict[str, int] = {}
        for i, model_group in enumerate(self._json["model_groups"]):
            for model_name in model_group["models"]:
                self._model_names_to_group_idx[model_name] = i

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
            raise ValueError(f"'{model_name}' is not defined")

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
        param_dict["target"] = tuple(param_dict["target"])
        param_dict["p_or_t"] = PlayerOrTeam(param_dict["p_or_t"]) if param_dict["p_or_t"] else None
        param_dict["data_filename"] = os.path.join(
            pathlib.Path(self._file_path).parent.as_posix(), param_dict["data_filename"]
        )

        if validation_failure_reason := typed_dict_validate(_Params, param_dict):
            raise ValueError(
                f"Model training parameter validation failure: {validation_failure_reason}"
            )
        return cast(_Params, param_dict)

    @process_timer
    def _train_and_test(
        self,
        model_name: str,
        automl_type: AutomlType,
        overwrite_mode: OverwriteMode,
        dest_dir: str | None,
        error_data: bool,
        **regressor_kwargs,
    ):
        params = self.get_params(model_name)

        # for any regressor kwarg not already set, fill in with model params
        if automl_type.startswith("tpot"):
            if not regressor_kwargs.get("random_state") and params["seed"]:
                regressor_kwargs["random_state"] = params["seed"]
            if params["train_params"]:
                if not set(params["train_params"].keys()) <= set(
                    _EXPECTED_TRAINING_PARAMS.__args__
                ):
                    raise ValueError(
                        "unexpected training parameters found in model definition file",
                        set(params["train_params"].keys())
                        - set(_EXPECTED_TRAINING_PARAMS.__args__),
                    )
                for arg in _EXPECTED_TRAINING_PARAMS.__args__:
                    if regressor_kwargs.get(arg) or not params["train_params"].get(arg):
                        continue
                    regressor_kwargs[arg] = params["train_params"][arg]

        print("Training will proceed with the following parameters:")
        pprint(params)
        print(f"Fitting model {model_name} with {automl_type} using {regressor_kwargs=}")

        raw_df, tt_data, one_hot_stats = load_data(
            params["data_filename"],
            params["target"],
            params["validation_season"],
            params["seed"],
            include_position=params["include_pos"],
            col_drop_filters=params["cols_to_drop"],
            missing_data_threshold=params["missing_data_threshold"],
            filtering_query=params["filtering_query"],
        )

        print(f"data load of '{params['data_filename']}' complete. {one_hot_stats=}")

        model = model_and_test(
            model_name,
            params["validation_season"],
            tt_data,
            params["target"],
            automl_type,
            params["p_or_t"],
            params["recent_games"],
            params["training_seasons"],
            regressor_kwargs,
            params["target_pos"],
            params["training_pos"] or params["target_pos"],
            dest_dir,
            raw_df=raw_df,
            overwrite_mode=overwrite_mode,
        )

        return model


_DEFAULT_AUTOML_TYPE: AutomlType = "tpot"
_DUMMY_REGRESSOR_KWARGS = {"strategy": "median"}


def main(cmd_line_str=None):
    parser = argparse.ArgumentParser(description="Train and Test CLI")
    parser.add_argument("train_file", help="Json file containing training parameters")
    parser.add_argument(
        "model",
        nargs="?",
        help="Name of the model to train, if not set then model names will be listed",
    )
    parser.add_argument("--info", default=False, action="store_true")
    parser.add_argument(
        "--overwrite_mode",
        "--mode",
        help="What action to take if a model already exists",
        default="fail",
        choices=OverwriteMode.__args__,
    )
    parser.add_argument(
        "--error_analysis_data",
        help="Write error analysis data based on validation dataset to a file. "
        "Data consists of columns 'truth', 'prediction', 'error'",
        default=False,
        action="store_true",
    )
    parser.add_argument("--automl_type", default=_DEFAULT_AUTOML_TYPE, choices=AutomlType.__args__)
    parser.add_argument(
        "--n_jobs", "--tpot_jobs", help="Number of jobs/processors to use during training", type=int
    )
    parser.add_argument(
        "--training_mins",
        "--mins",
        "--max_train_mins",
        "--time",
        "--max_time",
        "--max_time_mins",
        type=int,
        help="override the training time defined in the train_file",
    )
    parser.add_argument(
        "--training_iter_mins",
        "--max_iter_mins",
        "--iter_mins",
        "--iter_time",
        "--max_eval_time_mins",
        type=int,
        help="override the training iteration time defined in the train_file",
    )
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--dest_dir", default=".")

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    tdf = TrainingDefinitionFile(args.train_file)
    if not args.model:
        print(f"Following {len(tdf.model_names)} models are defined: {sorted(tdf.model_names)}")
        sys.exit(0)

    if args.model not in tdf.model_names:
        parser.error(
            f"Model '{args.model}' not defined. Try again with one "
            f"of the following: {tdf.model_names}"
        )

    if args.info:
        print(f"model parameters for {args.model}")
        pprint(tdf.get_params(args.model))
        sys.exit(0)

    if args.automl_type.startswith("tpot"):
        modeler_init_kwargs = {}
        if args.n_jobs:
            modeler_init_kwargs["n_jobs"] = args.n_jobs
        if args.training_mins:
            modeler_init_kwargs["max_time_mins"] = args.training_mins
        if args.training_iter_mins:
            modeler_init_kwargs["max_eval_time_mins"] = args.training_iter_mins
    elif args.automl_type == "dummy":
        modeler_init_kwargs = _DUMMY_REGRESSOR_KWARGS.copy()
    else:
        raise NotImplementedError()

    tdf._train_and_test(
        args.model,
        args.automl_type,
        args.overwrite_mode,
        args.dest_dir,
        args.error_analysis_data,
        **modeler_init_kwargs,
    )


if __name__ == "__main__":
    main()
