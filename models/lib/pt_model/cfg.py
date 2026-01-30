from __future__ import annotations

import json
import os
from functools import cache
from pprint import pprint
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast

import pandas as pd
from fantasy_py import (
    SPORT_DB_MANAGER_DOMAIN,
    CLSRegistry,
    FeatureType,
    InvalidArgumentsException,
    JSONWithCommentsDecoder,
    PlayerOrTeam,
    UnexpectedValueError,
    log,
)
from fantasy_py.inference import PTPredictModel, guess_sport_from_path
from ledona import process_timer
from typeguard import TypeCheckError, check_type

from .train_test import AlgorithmType, ModelFileFoundMode, load_data, model_and_test

if TYPE_CHECKING:
    from fantasy_py.sport import SportDBManager


_LOGGER = log.get_logger(__name__)


_NO_DEFAULT = object()
"""
used for training param default value when the regressor's default should be used.
This results in no kwarg being set for the regressor init param
"""


_TPOT_PARAM_DEFAULTS = {
    "max_time_mins": _NO_DEFAULT,
    "n_jobs": _NO_DEFAULT,
    "epochs_max": _NO_DEFAULT,  # epochs_max will be used for generations
    "early_stop": 3,
    "tp:max_eval_time_mins": _NO_DEFAULT,
    "tp:population_size": _NO_DEFAULT,
    # Following should have no impact on the resulting model
    "verbose": 3,
}
"""defaults for all tpot algorithms"""

TRAINING_PARAM_DEFAULTS: dict[AlgorithmType, dict[str, str | int | float | object]] = {
    "autogluon": {"max_time_mins": _NO_DEFAULT, "verbose": 2, "ag:preset": "best_v150"},
    "nn": {
        "epochs_max": _NO_DEFAULT,
        "early_stop": 20,
        "nn:hidden_size": _NO_DEFAULT,
        "nn:hidden_layers": _NO_DEFAULT,
        "nn:batch_size": _NO_DEFAULT,
        "nn:learning_rate": _NO_DEFAULT,
        "nn:shuffle": _NO_DEFAULT,
        "nn:resume_checkpoint_filepath": _NO_DEFAULT,
        "nn:checkpoint_dir": _NO_DEFAULT,
        "nn:checkpoint_frequency": _NO_DEFAULT,
    },
    "dummy": ({"dmy:strategy": "mean"}),
    "tpot": _TPOT_PARAM_DEFAULTS,
    "tpot-light": _TPOT_PARAM_DEFAULTS,
    "xgboost": {"verbose": 2},
}
"""
dict mapping algorithm to default-regressor-params

default-regressor-paras is a dict with all the regressor's supported
param names and default values. Each parameter corresponds to a train
cli parameter and a parameter used by the regressor for initialization
or fitting. Set a default parameter's value to _NO_DEFAULT to allow the
regressor to use its default.
"""


@cache
def _all_algo_params():
    """return a set of all valid params across all algorithms"""
    return {
        param
        for algo_defaults in TRAINING_PARAM_DEFAULTS.values()
        for param in algo_defaults.keys()
    }


"""set of all valid param names"""

_DATA_SRC_PARAMS = Literal["missing_data_threshold", "filtering_query", "data_filename"]
"""model parameters describing load and filtering of training data"""

DEFAULT_ALGORITHM: AlgorithmType = "tpot"

_IGNORE_ORIGINAL_PARAMS = {"resume_checkpoint_filepath"}
"""
model parameters that should not be reused on retrain, these are model parameters
and match parameter keys BEFORE name remappings
"""


class _TrainingParamsDict(TypedDict):
    """definition of the final parameters used for model training/testing"""

    data_filename: str
    sport: str
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
    train_params: dict[str, int | str | float | bool] | None
    """params passed to the training algorithm (likely as kwargs)"""
    original_model_columns: NotRequired[set[str]]
    """for use when retraining a model, the final input cols for the original model"""
    limit: int | None


class TrainingConfiguration:
    algorithm: AlgorithmType

    def __init__(
        self,
        filepath: str | None = None,
        cfg_dict: dict | None = None,
        cfg_dict_source: str | None = None,
        algorithm: AlgorithmType | None = None,
        retrain=False,
    ):
        """
        filepath: initialize using the contents of the json training configuration file
        cfg_dict: initialize using an existing configuration dict
        retrain: configuration is for retraining an existing model
        algorithm: if not explicitly defined then the global default at\
            DEFAULT_ALGORITHM is used.
        """
        if (filepath is None) == (cfg_dict is None):
            raise InvalidArgumentsException("filepath and cfg_dict cannot be both defined or None")
        if filepath is not None:
            self.source = filepath
            with open(filepath, "r") as f_:
                self._src_dict = cast(dict, json.load(f_, cls=JSONWithCommentsDecoder))
        else:
            assert cfg_dict_source is not None and isinstance(cfg_dict, dict)
            self.source = cfg_dict_source
            self._src_dict = cfg_dict

        sport = self._src_dict.get("sport")
        if sport is None:
            if filepath is not None:
                _LOGGER.warning(
                    "Sport was not found in '%s'. This must be an old model/cfg. "
                    "Attempting to infer sport from filename.",
                    filepath,
                )
                sport = guess_sport_from_path(filepath)
            if sport is None:
                raise InvalidArgumentsException(
                    "Unable to create training configuration. 'sport' key not found in "
                    "configuration and sport could not be inferred from filename!"
                )
        self.sport = sport

        self.retrain = retrain
        """retraining an existing model"""
        self.algorithm = algorithm or DEFAULT_ALGORITHM
        """training algorithm"""

        self._model_names_to_group_idx: dict[str, int] = {}
        for i, model_group in enumerate(self._src_dict["model_groups"]):
            for model_name in model_group["models"]:
                self._model_names_to_group_idx[model_name] = i

    @classmethod
    def cfg_from_model(
        cls,
        model_filepath: str,
        training_filepath: str | None,
        algorithm: AlgorithmType | None,
    ):
        """
        construct a model training definition that reflects the model

        parameters inheritance is prioritized (highest to lowest)
        1) model parameters in model_filepath
        2) parameters defined in the training definition file
        3) algorithm defaults

        model_filepath: path to the model that the new model should be based on
        training_filepath: path to the training definition file
        algorithm: algorithm of the new model, if None then retrain a model using the algorithm of the original model
        """
        orig_model = PTPredictModel.load(model_filepath)
        if orig_model.parameters is None or orig_model.performance is None:
            raise NotImplementedError(
                f"Model file for '{orig_model.name}' does not have parameters or performance. "
                "Retrain unsupported."
            )

        if algorithm is None:
            if "algorithm" not in orig_model.parameters:
                raise InvalidArgumentsException(
                    "'algorithm' is not present in the model definition "
                    "One must be provided (perhaps on the command line) to proceed"
                )
            algorithm = cast(AlgorithmType, orig_model.parameters["algorithm"])

        train_cfg_params = (
            TrainingConfiguration(training_filepath).get_params(orig_model.name)
            if training_filepath is not None
            else None
        )

        train_params = {}
        for param, default_value in TRAINING_PARAM_DEFAULTS[algorithm].items():
            if "." in param:
                raise UnexpectedValueError(
                    f"Algorithm '{algorithm}' training parameter name '{param}' is invalid. "
                    "Training parameter keys cannot contain '.'"
                )
            if param in orig_model.parameters and param not in _IGNORE_ORIGINAL_PARAMS:
                train_params[param] = orig_model.parameters[param]
                continue

            if train_cfg_params and param in train_cfg_params["train_params"]:
                train_params[param] = train_cfg_params["train_params"][param]
                continue

            if default_value != _NO_DEFAULT:
                train_params[param] = default_value

        model_params_dict = {
            key: orig_model.parameters[key]
            for key in _DATA_SRC_PARAMS.__args__
            if key in orig_model.parameters
        }
        model_params_dict.update(
            {
                "sport": orig_model.sport,
                "algorithm": algorithm,
                "target": orig_model.target.type + ":" + orig_model.target.name,
                "validation_season": orig_model.performance.get("season_val"),
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
                "limit": orig_model.data_def.get("limit"),
            }
        )

        cfg_dict: dict = {
            "sport": orig_model.sport,
            "global_default": {},
            "model_groups": [{"models": {orig_model.name: model_params_dict}}],
        }
        return (
            TrainingConfiguration(
                cfg_dict=cfg_dict,
                cfg_dict_source=model_filepath,
                retrain=True,
                algorithm=model_params_dict["algorithm"],
            ),
            orig_model,
        )

    @property
    def model_names(self):
        return list(self._model_names_to_group_idx.keys())

    def _params_from_cfg_levels(
        self, algo: str, model_name: str, global_cfg: dict, model_group_cfg: dict, model_cfg: dict
    ):
        """helper that returns training params for model_name
        from the global def dict"""
        all_valid_algorithm_params = _all_algo_params()

        global_cols_to_drop = global_cfg.get("cols_to_drop") or []
        global_train_params = global_cfg.get("train_params") or {}
        global_param_names = {name.split(".")[-1] for name in global_train_params}
        if invalids := global_param_names - all_valid_algorithm_params:
            raise InvalidArgumentsException(
                f"Invalid global training params found in {self.source}. {invalids=}"
            )

        group_cols_to_drop = model_group_cfg.get("cols_to_drop") or []
        group_train_params = model_group_cfg.get("train_params") or {}
        group_param_names = {name.split(".")[-1] for name in group_train_params}
        if invalids := group_param_names - all_valid_algorithm_params:
            raise InvalidArgumentsException(
                f"Invalid group training params found in '{self.source}' "
                f"group-name={self._model_names_to_group_idx[model_name]}. {invalids=}"
            )

        model_specific_cols_to_drop = model_cfg.get("cols_to_drop") or []
        model_specific_train_params = model_cfg.get("train_params") or {}
        model_param_names = {name.split(".")[-1] for name in model_specific_train_params}
        if invalids := model_param_names - all_valid_algorithm_params:
            raise InvalidArgumentsException(
                f"Invalid model training params found in '{self.source}' {model_name=}. {invalids=}"
            )

        final_train_params = {
            **global_train_params,
            **group_train_params,
            **model_specific_train_params,
        }

        # algo specific param handling
        algo_param_keys = [key for key in final_train_params if "." in key]
        for algo_param_key in algo_param_keys:
            param_algo, param_key = algo_param_key.split(".", 1)

            if param_algo != algo:
                # for a different algorithm
                del final_train_params[algo_param_key]
                continue

            # this param should be used!
            final_train_params[param_key] = final_train_params[algo_param_key]
            del final_train_params[algo_param_key]

        final_cols_to_drop = sorted(
            {
                *global_cols_to_drop,
                *group_cols_to_drop,
                *model_specific_cols_to_drop,
            }
        )

        return final_train_params, final_cols_to_drop

    @cache
    def get_params(self, model_name):
        """
        Return a dict containing the training/evaluation parameters
        for the requested model. Train parameters and cols_to_drop cascade.

        Train parameters are the union of parameters at all definition levels
        with the lower/more specific levels (model specific parameters are
        the most specific) taking precidence.

        cols_to_drop is the union of all cols_to_drop across all model
        definition levels.
        """
        if model_name not in self.model_names:
            raise UnexpectedValueError(f"'{model_name}' is not defined")

        # set everything that is None-able to None
        param_dict: dict = {
            param_key: None
            for param_key, value_type in _TrainingParamsDict.__annotations__.items()
            if hasattr(value_type, "__args__") and type(None) in value_type.__args__
        }
        param_dict["sport"] = self.sport
        if not self.retrain and "original_model_columns" in param_dict:
            del param_dict["original_model_columns"]

        global_cfg = self._src_dict["global_default"].copy()
        model_group_cfg = self._src_dict["model_groups"][self._model_names_to_group_idx[model_name]]
        model_cfg = model_group_cfg["models"][model_name]

        param_dict.update(self._src_dict["global_default"].copy())
        param_dict.update(
            {
                k_: v_
                for k_, v_ in model_group_cfg.items()
                if k_ not in ("cols_to_drop", "train_params", "models")
            }
        )

        final_train_params, final_cols_to_drop = self._params_from_cfg_levels(
            self.algorithm, model_name, global_cfg, model_group_cfg, model_cfg
        )
        param_dict.update(
            {k_: v_ for k_, v_ in model_cfg.items() if k_ not in ("cols_to_drop", "train_params")}
        )

        param_dict["train_params"] = final_train_params
        param_dict["cols_to_drop"] = final_cols_to_drop

        param_dict["target"] = (
            tuple(param_dict["target"])
            if isinstance(param_dict["target"], list)
            else param_dict["target"]
        )
        if param_dict["target"][0] == "extra":
            db_manager = cast(
                SportDBManager, CLSRegistry.get_class(SPORT_DB_MANAGER_DOMAIN, self.sport)
            )
            x_def = db_manager.EXTRA_STATS[param_dict["target"][1]]
            if x_def.get("current_type", "feature") != "target":
                raise InvalidArgumentsException(
                    f"model '{model_name}' has target '{param_dict['target']}' which, "
                    "based on its extra definition, cannot be targetted"
                )

        param_dict["p_or_t"] = PlayerOrTeam(param_dict["p_or_t"]) if param_dict["p_or_t"] else None
        param_dict["algorithm"] = self.algorithm

        try:
            check_type(param_dict, _TrainingParamsDict)
        except TypeCheckError as ex:
            raise UnexpectedValueError("Model training parameter validation failure") from ex
        if param_dict.get("training_pos") is None and param_dict.get("target_pos") is not None:
            param_dict["training_pos"] = param_dict["target_pos"]
        return cast(_TrainingParamsDict, param_dict)

    @staticmethod
    def _get_regressor_params(algorithm, cli_params: dict, model_params: dict):
        """
        Helper that finalizes regressor parameters based on the following (in order
        or precedence).

        1) cli/requested parameters
        2) config file model params
        3) algorithm defaults

        return the finalized regressor params
        """
        defaults = TRAINING_PARAM_DEFAULTS[algorithm]
        regressor_params: dict = {}

        if (
            "random_state" in defaults
            and cli_params.get("random_state") is None
            and model_params["seed"]
        ):
            regressor_params["random_state"] = model_params["seed"]

        for param_name in defaults:
            if (cli_value := cli_params.get(param_name)) is not None:
                regressor_params[param_name] = cli_value
                continue
            if (cfg_file_model_value := model_params["train_params"].get(param_name)) is not None:
                regressor_params[param_name] = cfg_file_model_value
                continue
            if (default_value := defaults[param_name]) != _NO_DEFAULT:
                regressor_params[param_name] = default_value
                continue

        if len(ignored_params := model_params["train_params"].keys() - defaults.keys()):
            _LOGGER.warning(
                "Ignoring following %i parameters not used by '%s' models: %s",
                len(ignored_params),
                algorithm,
                ignored_params,
            )

        return regressor_params

    @process_timer
    def train_and_test(
        self,
        model_name: str,
        dest_dir: str,
        file_found_mode: ModelFileFoundMode,
        data_dir: str | None,
        info: bool,
        dump_data: str,
        training_data_limit: None | int,
        dest_filename: str | None,
        **train_params,
    ):
        """
        train_params: these are all parameters submitted to define training. e.g. cli params in its entirety
        """
        params = self.get_params(model_name)
        limit = training_data_limit or params.get("limit")

        data_filepath = params["data_filename"]
        if data_dir is not None:
            data_filepath = os.path.join(data_dir, data_filepath)

        _LOGGER.info(
            "For %s loading data from '%s'%s",
            model_name,
            data_filepath,
            f" with limit {limit}" if limit else "",
        )

        target_tuple = cast(
            tuple[FeatureType, str],
            (
                params["target"]
                if isinstance(params["target"], (tuple, list))
                else params["target"].split(":")
            ),
        )
        if len(target_tuple) != 2 or (
            target_tuple[0] not in FeatureType.__args__ and target_tuple[0] != "extra"
        ):
            raise UnexpectedValueError(f"Invalid model target: {target_tuple}")

        final_regressor_params = self._get_regressor_params(
            self.algorithm, train_params, cast(dict, params)
        )
        print(f"\nInitial training params from '{self.source}' for '{model_name}':")
        pprint(params)
        print(f"\nFinal regressor kwargs for '{model_name}':")
        pprint(final_regressor_params)
        if limit is not None:
            print(f"with a training data limit of {limit}")

        try:
            _, tt_data, one_hot_stats = load_data(
                data_filepath,
                target_tuple,
                params["validation_season"],
                params["seed"],
                include_position=params["include_pos"],
                col_drop_filters=params["cols_to_drop"],
                missing_data_warn_threshold=params.get("missing_data_threshold", 0),
                filtering_query=params["filtering_query"],
                limit=limit,
                expected_cols=params["original_model_columns"] if self.retrain else None,
            )
            _LOGGER.info(
                "for %s data load of '%s' complete. one_hot_stats=%s",
                model_name,
                params["data_filename"],
                one_hot_stats,
            )
            if dump_data:
                _LOGGER.info("Dumping training data to '%s'", dump_data)
                df = pd.concat(tt_data[0:2], axis=1)
                if dump_data.endswith(".csv"):
                    df.to_csv(dump_data)
                elif dump_data.endswith(".parquet"):
                    df.to_parquet(dump_data)
                else:
                    raise UnexpectedValueError(f"Unknown data dump format: {dump_data}")
        except FileNotFoundError:
            if not info:
                raise
            _LOGGER.warning("Data file '%s' was not found", data_filepath)

        if info:
            return None

        data_src_params: dict[_DATA_SRC_PARAMS, str | float | None] = {
            "missing_data_threshold": params.get("missing_data_threshold", 0),
            "filtering_query": params["filtering_query"],
            "data_filename": params["data_filename"],
        }

        model = model_and_test(
            model_name,
            params["sport"],
            params["validation_season"],
            tt_data,
            target_tuple,
            self.algorithm,
            params["p_or_t"],
            params["recent_games"],
            params["training_seasons"],
            final_regressor_params,
            params["target_pos"],
            params["training_pos"] or params["target_pos"],
            dest_dir,
            file_found_mode,
            limit,
            model_dest_filename=dest_filename,
            data_src_params=data_src_params,
        )

        return model
