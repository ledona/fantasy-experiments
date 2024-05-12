import json
import os
import sys
import warnings
from pprint import pprint
from typing import Literal, TypedDict, cast

import pandas as pd
from fantasy_py import (
    FeatureType,
    InvalidArgumentsException,
    JSONWithCommentsDecoder,
    PlayerOrTeam,
    UnexpectedValueError,
    log,
    typed_dict_validate,
)
from fantasy_py.inference import PTPredictModel

from .train_test import AlgorithmType, load_data, model_and_test

_LOGGER = log.get_logger(__name__)

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

_TPOT_TRAINING_PARAMS = Literal[
    "max_time_mins", "max_eval_time_mins", "n_jobs", "epochs_max", "early_stop"
]
"""names of parameters in a model definition file applicable to tpot"""

_TPOT_TRAINING_PARAMS_RENAME = {"epochs_max": "generations"}
"""remapping of model parameter names in definition file to tpot kwarg parameter names"""

_DUMMY_TRAINING_PARAMS = Literal["strategy"]

_DATA_SRC_PARAMS = Literal["missing_data_threshold", "filtering_query", "data_filename"]
"""model parameters describing load and filtering of training data"""

DEFAULT_ALGORITHM: AlgorithmType = "tpot"


def _get_param_keys(algorithm: AlgorithmType) -> tuple[set[str], None | dict[str, str]]:
    """
    returns tuple[expected-algorithm-param-keys, key-renamer-dict] where the expected keys
    are the regressor instantiation kw args required by the algorithm, and
    the renamer is a mapping of the expected key to the actual kwarg name used
    when instantiating the regressor. Default (if the expected key is not in the renamer)
    is to use the expected key name itself. The renamer helps with the reuse of expected
    key names by multiple algorithms
    """
    if algorithm not in AlgorithmType.__args__:
        raise UnexpectedValueError(f"Param keys unknown for {algorithm=}")
    if algorithm.startswith("tpot"):
        return set(_TPOT_TRAINING_PARAMS.__args__), _TPOT_TRAINING_PARAMS_RENAME
    if algorithm == "nn":
        return set(_NN_TRAINING_PARAMS.__args__), None
    if algorithm == "dummy":
        return set(_DUMMY_TRAINING_PARAMS.__args__), None
    return set(), None


class _TrainingParamsDict(TypedDict):
    """definition of the final parameters used for model training/testing"""

    data_filename: str
    target: tuple[Literal["stat", "calc", "extra"], str] | str
    """target stat, either a tuple of (type, name) or string of 'type:name'"""
    validation_season: int
    recent_games: int
    training_seasons: list[int]
    algorithm: AlgorithmType

    # nullable/optional values
    seed: None | int
    p_or_t: PlayerOrTeam | None
    include_pos: bool | None
    cols_to_drop: list[str] | None
    """columns/features to drop from training data. wildcards/regexs are accepted
    must be None if cols_to_drop is not None"""
    missing_data_threshold: float | None
    filtering_query: str | None
    """pandas compatible query string that will be run on the loaded data"""
    target_pos: list[str] | None
    training_pos: list[str] | None
    train_params: dict[_TPOT_TRAINING_PARAMS | _NN_TRAINING_PARAMS, int | str | float | bool] | None
    """params passed to the training algorithm (likely as kwargs)"""
    original_model_columns: set[str] | None
    """for use when retraining a model, the final input cols for the original model"""


class TrainingConfiguration:
    def __init__(
        self,
        filepath: str | None = None,
        cfg_dict: dict | None = None,
        algorithm: AlgorithmType | None = None,
        retrain=False,
    ):
        """
        filepath: initialize using the contents of the json training configuration file
        cfg_dict: initialize using an existing configuration dict
        retrain: configuration is for retraining an existing model
        algorithm: default is DEFAULT_ALGORITHM
        """
        if (filepath is None) == (cfg_dict is None):
            raise InvalidArgumentsException("filepath and cfg_dict cannot be both defined or None")
        if filepath is not None:
            with open(filepath, "r") as f_:
                self._json = cast(dict, json.load(f_, cls=JSONWithCommentsDecoder))
        else:
            assert cfg_dict is not None
            self._json = cfg_dict

        self.retrain = retrain
        """retraining an existing model"""
        self.algorithm = algorithm or DEFAULT_ALGORITHM
        """training algorithm"""

        self._model_names_to_group_idx: dict[str, int] = {}
        for i, model_group in enumerate(self._json["model_groups"]):
            for model_name in model_group["models"]:
                self._model_names_to_group_idx[model_name] = i

    # TODO: remove training_filepath asap
    @classmethod
    def cfg_from_model(
        cls,
        model_filepath: str,
        training_filepath: str | None,
        algorithm: AlgorithmType | None,
    ):
        """
        create a model training definition that reflects the model
        algorithm: if None then retrain using the same algorithm as the original model
        """
        orig_model = PTPredictModel.load(model_filepath)
        if orig_model.parameters is None or orig_model.performance is None:
            raise NotImplementedError(
                f"Model file for '{orig_model.name}' does not have parameters or performance. "
                "Retrain unsupported."
            )

        param_keys, _ = _get_param_keys(orig_model.parameters["algorithm"])
        train_params = (
            {key: orig_model.parameters[key] for key in param_keys}
            if param_keys is not None
            else None
        )
        model_params_dict = {
            key: orig_model.parameters[key]
            for key in _DATA_SRC_PARAMS.__args__
            if key in orig_model.parameters
        }
        model_params_dict.update(
            {
                "algorithm": algorithm or orig_model.parameters["algorithm"],
                "target": orig_model.target.type + ":" + orig_model.target.name,
                "validation_season": cast(int, orig_model.performance["season"]),
                "recent_games": orig_model.data_def["recent_games"],
                "training_seasons": orig_model.data_def["seasons"],
                "seed": orig_model.parameters.get("random_state"),
                "p_or_t": orig_model.target.p_or_t,
                "include_pos": orig_model.data_def["include_pos"],
                "cols_to_drop": None,
                "original_model_columns": set(orig_model.data_def["input_cols"]),
                "target_pos": orig_model.player_positions,
                "training_pos": orig_model.player_positions,
                "train_params": train_params,
            }
        )

        missing_keys = set(_TrainingParamsDict.__annotations__.keys()) - set(
            model_params_dict.keys()
        )
        if len(missing_keys) > 0:
            if training_filepath is None:
                raise UnexpectedValueError(
                    f"Training parameters for model '{orig_model.name}' in '{model_filepath}' "
                    f"are incomplete. A training cfg file is required! "
                    "missing_keys={sorted(missing_keys)}"
                )
            warnings.warn(
                "Using training cfg file as a fallback for a model with missing parameters "
                "will be dropped in future versions.",
                DeprecationWarning,
            )

            _LOGGER.info(
                "Retrieving the following missing training parameters for '%s' from '%s': %s",
                orig_model.name,
                training_filepath,
                missing_keys,
            )
            config = TrainingConfiguration(training_filepath)
            model_params = config.get_params(orig_model.name)
            for key in missing_keys:
                model_params_dict[key] = model_params[key]

            if orig_model.target[0] + ":" + orig_model.target[2] != model_params_dict["target"]:
                raise UnexpectedValueError(
                    f"target of new configuration is {model_params_dict['target']} does not match "
                    "old model target {model.target} "
                )

        cfg_dict: dict = {
            "global_default": {},
            "model_groups": [{"models": {orig_model.name: model_params_dict}}],
        }
        return (
            TrainingConfiguration(
                cfg_dict=cfg_dict, retrain=True, algorithm=model_params_dict["algorithm"]
            ),
            orig_model,
        )

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
            for param_key, value_type in _TrainingParamsDict.__annotations__.items()
            if hasattr(value_type, "__args__") and type(None) in value_type.__args__
        }
        if not self.retrain:
            del param_dict["original_model_columns"]

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
            param_dict["train_params"] = (
                {} if not param_dict["train_params"] else param_dict["train_params"].copy()
            )
            param_dict["train_params"].update(model_train_params)
        param_dict["target"] = (
            tuple(param_dict["target"])
            if isinstance(param_dict["target"], list)
            else param_dict["target"]
        )
        param_dict["p_or_t"] = PlayerOrTeam(param_dict["p_or_t"]) if param_dict["p_or_t"] else None
        param_dict["algorithm"] = self.algorithm

        if validation_failure_reason := typed_dict_validate(_TrainingParamsDict, param_dict):
            raise UnexpectedValueError(
                f"Model training parameter validation failure: {validation_failure_reason}"
            )
        if param_dict.get("training_pos") is None and param_dict.get("target_pos") is not None:
            param_dict["training_pos"] = param_dict["target_pos"]
        return cast(_TrainingParamsDict, param_dict)

    def _get_regressor_kwargs(self, regressor_kwargs: dict, params: dict):
        # for any regressor kwarg not already set, fill in with model params
        try:
            expected_param_names, renamer = _get_param_keys(self.algorithm)
        except UnexpectedValueError as ex:
            _LOGGER.warning(
                "Failed to get expected param keys, falling back on model params", exc_info=ex
            )
            return regressor_kwargs

        new_kwargs = {
            key: value for key, value in regressor_kwargs.items() if key in expected_param_names
        }
        if (
            "random_state" in expected_param_names
            and new_kwargs.get("random_state") is None
            and params["seed"]
        ):
            key = renamer.get("random_state", "random_state") if renamer else "random_state"
            new_kwargs[key] = params["seed"]
        if params["train_params"]:
            if not set(params["train_params"].keys()) <= expected_param_names:
                _LOGGER.warning(
                    "Ignoring following parameters not used by '%s' models: %s",
                    self.algorithm,
                    set(params["train_params"].keys()) - expected_param_names,
                )
            for arg in expected_param_names:
                if new_kwargs.get(arg) or not params["train_params"].get(arg):
                    continue
                name = renamer.get(arg, arg) if renamer else arg
                new_kwargs[name] = params["train_params"][arg]

        return new_kwargs

    def _train_and_test(
        self,
        model_name: str,
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

        target_tuple = cast(
            tuple[FeatureType, str],
            (
                params["target"]
                if isinstance(params["target"], (tuple, list))
                else params["target"].split(":")
            ),
        )
        if len(target_tuple) != 2 or target_tuple[0] not in FeatureType.__args__:
            raise UnexpectedValueError(f"Invalid model target: {target_tuple}")

        _, tt_data, one_hot_stats = load_data(
            data_filepath,
            target_tuple,
            params["validation_season"],
            params["seed"],
            include_position=params["include_pos"],
            col_drop_filters=params["cols_to_drop"],
            missing_data_threshold=params.get("missing_data_threshold", 0),
            filtering_query=params["filtering_query"],
            limit=limit,
            expected_cols=params["original_model_columns"] if self.retrain else None,
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

        final_regressor_kwargs = self._get_regressor_kwargs(regressor_kwargs, params)
        print("\nTraining will proceed with the following parameters:")
        pprint(params)
        print()
        _LOGGER.info(
            "Fitting model '%s' with algorithm=%s using final_regressor_kwargs=%s",
            model_name,
            self.algorithm,
            final_regressor_kwargs,
        )

        if info:
            print(f"\nModel parameters for {model_name}")
            pprint(params)
            print(f"Data features (n={len(tt_data[0].columns)}): {sorted(tt_data[0].columns)}")
            sys.exit(0)

        data_src_params: dict[_DATA_SRC_PARAMS, str | float | None] = {
            "missing_data_threshold": params.get("missing_data_threshold", 0),
            "filtering_query": params["filtering_query"],
            "data_filename": params["data_filename"],
        }

        model = model_and_test(
            model_name,
            params["validation_season"],
            tt_data,
            target_tuple,
            self.algorithm,
            params["p_or_t"],
            params["recent_games"],
            params["training_seasons"],
            final_regressor_kwargs,
            params["target_pos"],
            params["training_pos"] or params["target_pos"],
            dest_dir,
            reuse_existing_models,
            model_dest_filename=dest_filename,
            data_src_params=data_src_params,
        )

        return model
