import json
import os
from pprint import pprint
from typing import Literal, TypedDict, cast

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
    typed_dict_validate,
)
from fantasy_py.inference import PTPredictModel, guess_sport_from_path
from fantasy_py.sport import SportDBManager
from ledona import process_timer

from .train_test import AlgorithmType, ModelFileFoundMode, load_data, model_and_test

_LOGGER = log.get_logger(__name__)


_NO_DEFAULT = object()
"""
used for training param default value when the regressor's default should be used.
This results in no kwarg being set for the regressor init param
"""


_TPOT_PARAM_DEFAULTS_TUPLE = {
    "max_time_mins": _NO_DEFAULT,
    "max_eval_time_mins": _NO_DEFAULT,
    "n_jobs": _NO_DEFAULT,
    "epochs_max": _NO_DEFAULT,
    "early_stop": _NO_DEFAULT,
    "population_size": _NO_DEFAULT,
    # Following should have no impact on the resulting model
    "use_dask": _NO_DEFAULT,
    "verbosity": _NO_DEFAULT,
}, {"epochs_max": "generations"}
"""defaults and renames for all tpot algorithms"""

TRAINING_PARAM_DEFAULTS: dict[AlgorithmType, tuple[dict, dict | None]] = {
    "nn": (
        {
            "input_size": _NO_DEFAULT,
            "hidden_size": _NO_DEFAULT,
            "hidden_layers": _NO_DEFAULT,
            "batch_size": _NO_DEFAULT,
            "epochs_max": _NO_DEFAULT,
            "early_stop": _NO_DEFAULT,
            "learning_rate": _NO_DEFAULT,
            "shuffle": _NO_DEFAULT,
            "resume_checkpoint_filepath": _NO_DEFAULT,
            "checkpoint_dir": _NO_DEFAULT,
            "checkpoint_frequency": _NO_DEFAULT,
        },
        {"early_stop": "early_stop_epochs"},
    ),
    "dummy": ({"strategy": "mean"}, None),
    "tpot": _TPOT_PARAM_DEFAULTS_TUPLE,
    "tpot-light": _TPOT_PARAM_DEFAULTS_TUPLE,
    "tpot-xgboost": _TPOT_PARAM_DEFAULTS_TUPLE,
    "xgboost": ({}, None)
}
"""
dict mapping algorithm to (default-regressor-params, param-rename-dict)

default-regressor-paras : A dict with all the regressor parameters and default
  values for the algorithm. Each parameter corresponds to a keyword arg that
  is used to instantiate the regressor object for the algorithm. 
  Using the value __NO_DEFAULT will cause that keyword arg to not be specified,
  thereby allowing the regressor to use its default.

param-rename-dict : This allows different default dicts to share parameter names
  for parameters that have the same effect on different regressors, even if
  the regressor object uses a different name for that parameter's kwarg during
  regressor instantiation.
"""

_DATA_SRC_PARAMS = Literal["missing_data_threshold", "filtering_query", "data_filename"]
"""model parameters describing load and filtering of training data"""

DEFAULT_ALGORITHM: AlgorithmType = "tpot"

_IGNORE_ORIGINAL_PARAMS = {"resume_checkpoint_filepath"}
"""
model parameters that should not be reused on retrain, these are model parameters
and match parameter keys BEFORE name remappings, use the rename dict from default
if after name remapping parameter keys are needed
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
    original_model_columns: set[str] | None
    """for use when retraining a model, the final input cols for the original model"""


class TrainingConfiguration:
    algorithm: AlgorithmType

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
        algorithm: if not explicitly defined then the global default at\
            DEFAULT_ALGORITHM is used.
        """
        if (filepath is None) == (cfg_dict is None):
            raise InvalidArgumentsException("filepath and cfg_dict cannot be both defined or None")
        if filepath is not None:
            with open(filepath, "r") as f_:
                self._json = cast(dict, json.load(f_, cls=JSONWithCommentsDecoder))
        else:
            assert cfg_dict is not None
            self._json = cfg_dict

        sport = self._json.get("sport")
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
        for i, model_group in enumerate(self._json["model_groups"]):
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
        algorithm: if None then retrain using the same algorithm as the original model
        """
        orig_model = PTPredictModel.load(model_filepath)
        if orig_model.parameters is None or orig_model.performance is None:
            raise NotImplementedError(
                f"Model file for '{orig_model.name}' does not have parameters or performance. "
                "Retrain unsupported."
            )

        if algorithm is None:
            if "algorithm" not in orig_model.parameters:
                _LOGGER.warning(
                    "algorithm was not found in the original model's "
                    "parameters, attempting to infer from model filename"
                )
                for algo in AlgorithmType.__args__:
                    if f".{algo}." in model_filepath:
                        algorithm = algo
                        _LOGGER.info(
                            "Based on model filename, will proceed with algorithm='%s'", algorithm
                        )
                        break
                if algorithm is None:
                    raise UnexpectedValueError(
                        "'algorithm' is not present in the model definition "
                        "and could not be inferred from model filename. "
                        "One must be provided (perhaps on the command line) to proceed"
                    )
            else:
                algorithm = cast(AlgorithmType, orig_model.parameters["algorithm"])
        defaults, renamer = TRAINING_PARAM_DEFAULTS[algorithm]

        train_params = {}
        for key in defaults:
            model_key = renamer.get(key, key) if renamer else key
            if model_key in orig_model.parameters and key not in _IGNORE_ORIGINAL_PARAMS:
                train_params[key] = orig_model.parameters[model_key]
                continue
            default_value = defaults[key]
            if default_value == _NO_DEFAULT:
                continue
            train_params[key] = default_value

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
                "validation_season": orig_model.performance["season_val"],
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

        missing_model_param_keys = set(_TrainingParamsDict.__annotations__.keys()) - set(
            model_params_dict.keys()
        )
        missing_train_param_keys = (
            set(defaults.keys()) - set(train_params.keys()) - _IGNORE_ORIGINAL_PARAMS
        )

        if len(missing_model_param_keys) > 0 or len(missing_train_param_keys) > 0:
            train_cfg_params = (
                TrainingConfiguration(training_filepath).get_params(orig_model.name)
                if training_filepath is not None
                else None
            )

            if len(missing_model_param_keys) > 0:
                # this should only happen when required common model parameters were not
                # being saved to .model files at the time the original model was created.
                if train_cfg_params is None:
                    raise UnexpectedValueError(
                        f"Modeling parameters for original model '{orig_model.name}' in "
                        f"'{model_filepath}' are incomplete. This is caused by the .model "
                        "being out of date and not including all commonly required "
                        "parameters. A training cfg file is required! "
                        f"missing_model_params={sorted(missing_model_param_keys)}"
                    )
                _LOGGER.info(
                    "Retrieving the following missing model parameters for '%s' from '%s': %s",
                    orig_model.name,
                    training_filepath,
                    missing_model_param_keys,
                )
                for key in missing_model_param_keys:
                    model_params_dict[key] = train_cfg_params[key]

                if orig_model.target[0] + ":" + orig_model.target[2] != model_params_dict["target"]:
                    raise UnexpectedValueError(
                        f"target of new configuration is {model_params_dict['target']} does not "
                        f"match old model target {orig_model.target}"
                    )

            if len(missing_train_param_keys) > 0:
                if train_cfg_params is not None:
                    assert train_cfg_params["train_params"]
                    for key in missing_train_param_keys:
                        if key not in train_cfg_params["train_params"]:
                            continue
                        model_params_dict["train_params"][key] = train_cfg_params["train_params"][
                            key
                        ]

        cfg_dict: dict = {
            "sport": orig_model.sport,
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
        if model_name not in self.model_names:
            raise UnexpectedValueError(f"'{model_name}' is not defined")

        # set everything that is Noneable to None
        param_dict: dict = {
            param_key: None
            for param_key, value_type in _TrainingParamsDict.__annotations__.items()
            if hasattr(value_type, "__args__") and type(None) in value_type.__args__
        }
        param_dict["sport"] = self.sport
        if not self.retrain:
            del param_dict["original_model_columns"]

        param_dict.update(self._json["global_default"].copy())

        model_group = self._json["model_groups"][self._model_names_to_group_idx[model_name]]
        param_dict.update(
            {k_: v_ for k_, v_ in model_group.items() if k_ not in ("train_params", "models")}
        )

        param_dict.update(
            {k_: v_ for k_, v_ in model_group["models"][model_name].items() if k_ != "train_params"}
        )

        train_params: dict = (
            param_dict["train_params"].copy() if param_dict.get("train_params") else {}
        )
        if model_group_train_params := model_group.get("train_params"):
            train_params.update(model_group_train_params)
        if model_train_params := model_group["models"][model_name].get("train_params"):
            train_params.update(model_train_params)
        param_dict["train_params"] = train_params

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

        if validation_failure_reason := typed_dict_validate(_TrainingParamsDict, param_dict):
            raise UnexpectedValueError(
                f"Model training parameter validation failure: {validation_failure_reason}"
            )
        if param_dict.get("training_pos") is None and param_dict.get("target_pos") is not None:
            param_dict["training_pos"] = param_dict["target_pos"]
        return cast(_TrainingParamsDict, param_dict)

    @staticmethod
    def _get_regressor_kwargs(algorithm, cli_regressor_params: dict, cfg_params: dict):
        """
        helper that finalizes regressor kwargs based on algorith, requested kwargs
        (e.g. from command line) and config parameters from
        regressor_kwargs: override regressor kwargs, likely defined on the command line
        params: default training parameters, typically specified in the
        return the finalized regressor kwargs
        """
        defaults, renamer = TRAINING_PARAM_DEFAULTS[algorithm]
        new_kwargs: dict = {}

        if (
            "random_state" in defaults
            and new_kwargs.get("random_state") is None
            and cfg_params["seed"]
        ):
            key = renamer.get("random_state", "random_state") if renamer else "random_state"
            new_kwargs[key] = cfg_params["seed"]

        for k_, default_value in defaults.items():
            key = renamer.get(k_, k_) if renamer is not None else k_
            if key in cli_regressor_params:
                new_kwargs[key] = cli_regressor_params[key]
                continue
            if default_value != _NO_DEFAULT:
                new_kwargs[key] = default_value
                continue
            if not cfg_params["train_params"] or k_ not in cfg_params["train_params"]:
                continue
            new_kwargs[key] = cfg_params["train_params"][k_]

        if cfg_params["train_params"] and not set(cfg_params["train_params"].keys()) <= (
            default_param_names := set(defaults.keys())
        ):
            _LOGGER.warning(
                "Ignoring following parameters not used by '%s' models: %s",
                algorithm,
                set(cfg_params["train_params"].keys()) - default_param_names,
            )

        assert set(new_kwargs.keys()) <= {
            renamer.get(name, name) if renamer else name for name in defaults
        }, "finalized kwargs should be a subset of the defaults"
        return new_kwargs

    @process_timer
    def train_and_test(
        self,
        model_name: str,
        dest_dir: str,
        error_data: bool,
        file_found_mode: ModelFileFoundMode,
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

        final_regressor_kwargs = self._get_regressor_kwargs(
            self.algorithm, regressor_kwargs, cast(dict, params)
        )
        print(f"\nTraining of {model_name} will be based on the following parameters:")
        pprint(params)
        print()
        print(
            f"Training of {model_name} will proceed with the regressor "
            "kwargs (overriding any previous params):"
        )
        pprint(final_regressor_kwargs)

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
            final_regressor_kwargs,
            params["target_pos"],
            params["training_pos"] or params["target_pos"],
            dest_dir,
            file_found_mode,
            model_dest_filename=dest_filename,
            data_src_params=data_src_params,
        )

        return model
