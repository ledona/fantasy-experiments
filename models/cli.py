#! /venv/bin/python
import argparse
import json
import os
import pathlib
import sys
from pprint import pprint
from typing import Literal, TypedDict, cast

from fantasy_py import PlayerOrTeam, typed_dict_validate
from ledona import process_timer

sys.path.append("..")
from train_test import load_data, model_and_test, AutomlType


class _Params(TypedDict):
    """definition of the final parameters used for model training/testing"""

    data_filename: str
    target: tuple[Literal["stat", "calc"], str]
    validation_season: int
    training_time: int
    """maximum time (seconds) for full training"""
    p_or_t: PlayerOrTeam
    recent_games: int
    training_seasons: list[int]

    include_pos: bool | None
    cols_to_drop: list[str] | None
    seed: int | None
    missing_data_threshold: float | None
    filtering_query: str | None
    target_pos: list[str] | None
    training_pos: list[str] | None
    iteration_time: int | None
    """maximum time (seconds) for a single iteration of training"""


class TrainingDefinitionFile:
    def __init__(self, file_path: str):
        self._file_path = file_path
        with open(file_path, "r") as f_:
            self._json: dict = json.load(f_)

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

        param_dict.update(self._json["global_defaults"].copy())
        if model_name not in self.model_names:
            raise ValueError(f"'{model_name}' is not defined")

        model_group = self._json["model_groups"][self._model_names_to_group_idx[model_name]]
        param_dict.update({k_: v_ for k_, v_ in model_group.items() if k_ != "models"})
        param_dict.update(model_group["models"][model_name])
        param_dict["target"] = tuple(param_dict["target"])
        param_dict["p_or_t"] = PlayerOrTeam(param_dict["p_or_t"])
        param_dict["data_filename"] = os.path.join(
            pathlib.Path(self._file_path).parent.as_posix(), param_dict["data_filename"]
        )

        if validation_failure_reason := typed_dict_validate(_Params, param_dict):
            raise ValueError(
                f"Model training parameter validation failure: {validation_failure_reason}"
            )
        return cast(_Params, param_dict)

    @process_timer
    def train_and_test(
        self,
        model_name: str,
        train_secs_override: int | None,
        train_iter_secs_override: int | None,
        reuse_existing: bool,
        automl_type: str,
        tpot_jobs: int,
    ):
        params = self.get_params(model_name)

        print("Training will proceed with the following parameters:")
        pprint(params)
        print()

        raw_df, tt_data, one_hot_stats = load_data(
            params["data_filename"],
            params["target"],
            params["validation_season"],
            include_position=params["include_pos"],
            col_drop_filters=params["cols_to_drop"],
            seed=params["seed"],
            missing_data_threshold=params["missing_data_threshold"],
            filtering_query=params["filtering_query"],
        )

        print(f"data load of '{params['data_filename']}' complete. {one_hot_stats=}")

        model = model_and_test(
            model_name,
            params["validation_season"],
            tt_data,
            params["target"],
            train_secs_override or params["training_time"],
            automl_type,
            params["p_or_t"],
            params["recent_games"],
            params["training_seasons"],
            seed=params["seed"],
            target_pos=params["target_pos"],
            training_pos=params["target_pos"],
            raw_df=raw_df,
            reuse_existing=reuse_existing,
            automl_kwargs={
                "n_jobs": tpot_jobs,
                "max_eval_time_mins": train_iter_secs_override or params["iteration_time"],
            },
        )

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test CLI")
    parser.add_argument("train_file", help="Json file containing training parameters")
    parser.add_argument(
        "model",
        nargs="?",
        help="Name of the model to train, if not set then model names will be listed",
    )
    parser.add_argument("--info", default=False, action="store_true")
    parser.add_argument(
        "--reuse_existing",
        default=False,
        action="store_true",
        help="If an existing model exists at the destination then load and "
        "evalute that instead of training a fresh model",
    )
    parser.add_argument("--automl_type", default="tpot", choices=AutomlType.__args__)
    parser.add_argument("--tpot_jobs", type=int, default=2)
    parser.add_argument(
        "--training_mins",
        "--mins",
        "--time",
        type=int,
        help="override the training time defined in the train_file",
    )
    parser.add_argument(
        "--training_iter_mins",
        "--iter_mins",
        "--iter_time",
        type=int,
        help="override the training iteration time defined in the train_file",
    )
    args = parser.parse_args()

    tdf = TrainingDefinitionFile(args.train_file)
    if not args.model:
        print(f"Following {len(tdf.model_names)} models are defined: {sorted(tdf.model_names)}")
        sys.exit(0)

    if args.model not in tdf.model_names:
        parser.error(
            f"Model '{args.model}' not defined. Try again with one "
            "of the following: {tdf.model_names}"
        )

    if args.info:
        print(f"model parameters for {args.model}")
        pprint(tdf.get_params(args.model))
        sys.exit(0)

    tdf.train_and_test(
        args.model,
        args.training_mins * 60 if args.training_mins else None,
        args.training_iter_mins * 60 if args.training_iter_mins else None,
        args.reuse_existing,
        args.automl_type,
        args.tpot_jobs,
    )
