import json
import os
import platform
import random
from datetime import datetime
from typing import cast

import joblib
import pandas as pd
import pytest
import sklearn
from fantasy_py import FeatureType, PlayerOrTeam, dt_to_filename_str
from fantasy_py.inference import PTPredictModel
from freezegun import freeze_time
from ledona import deep_compare_dicts
from pytest_mock import MockFixture
from sklearn.dummy import DummyRegressor

from ..pt_model import (
    DEFAULT_ALGORITHM,
    TRAINING_PARAM_DEFAULTS,
    AlgorithmType,
    TrainingConfiguration,
    _TrainingParamsDict,
)
from ..pt_model.cfg import _DATA_SRC_PARAMS, _NO_DEFAULT
from ..regressor import main

_EXPECTED_TRAINING_CFG_PARAMS = {
    "MLB-team-runs": {
        "seed": 1,
        "p_or_t": PlayerOrTeam.TEAM,
        "cols_to_drop": [
            "extra:(whip|venue|opp|hit|is_home).*",
            "stat:.*:.*",
            ".*opp-team",
            ".*:p.*",
        ],
        "missing_data_threshold": 0.1,
        "train_params": {
            "epochs_max": 100,
            "early_stop": 5,
            "max_eval_time_mins": 15,
            "max_time_mins": 60,
        },
        "validation_season": 2023,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_team.pq",
        "target": "stat:off_runs",
    },
    "MLB-P-DK": {
        "seed": 1,
        "p_or_t": PlayerOrTeam.PLAYER,
        "cols_to_drop": [".*y_score.*", ".*recent-.*", "extra:venue.*", ".*dk_score:.*"],
        "missing_data_threshold": 0.28,
        "train_params": {
            "epochs_max": 100,
            "early_stop": 5,
            "max_eval_time_mins": 15,
            "max_time_mins": 60,
        },
        "include_pos": False,
        "validation_season": 2023,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_pitcher.pq",
        "target": "calc:dk_score",
        "target_pos": ["P"],
    },
    "MLB-H-DK": {
        "seed": 1,
        "p_or_t": PlayerOrTeam.PLAYER,
        "cols_to_drop": ["extra:bases", ".*y_score.*"],
        "missing_data_threshold": 0.1,
        "train_params": {
            "batch_size": 64,
            "hidden_layers": 1,
            "epochs_max": 100,
            "early_stop": 5,
            "max_eval_time_mins": 15,
            "max_time_mins": 45,
            "n_jobs": 2,
        },
        "validation_season": 2023,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_hitter.pq",
        "target": "calc:dk_score",
        "include_pos": True,
        "target_pos": ["1B", "2B", "3B", "SS", "C", "LF", "RF", "CF", "OF"],
    },
    "MLB-H-hit": {
        "seed": 1,
        "p_or_t": PlayerOrTeam.PLAYER,
        "cols_to_drop": ["extra:bases"],
        "missing_data_threshold": 0.1,
        "train_params": {
            "epochs_max": 100,
            "early_stop": 5,
            "max_eval_time_mins": 15,
            "max_time_mins": 45,
            "n_jobs": 2,
        },
        "validation_season": 2023,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_hitter.pq",
        "target": "stat:off_hit",
        "include_pos": True,
        "target_pos": ["1B", "2B", "3B", "SS", "C", "LF", "RF", "CF", "OF"],
    },
}

_DIRNAME = os.path.dirname(os.path.abspath(__file__))

# add keys for Noneable parameters
for model_params in _EXPECTED_TRAINING_CFG_PARAMS.values():
    if model_params.get("training_pos") is None and model_params.get("target_pos") is not None:
        model_params["training_pos"] = model_params.get("target_pos")
    pad_dict = {
        param_key: None
        for param_key, value_type in _TrainingParamsDict.__annotations__.items()
        if hasattr(value_type, "__args__")
        and type(None) in value_type.__args__
        and param_key not in model_params
        and param_key != "original_model_columns"
    }
    model_params.update(pad_dict)


_TEST_DEF_FILE_FILEPATH = os.path.join(_DIRNAME, "test.json")


@pytest.fixture(name="tdf", scope="module")
def _tdf():
    return TrainingConfiguration(_TEST_DEF_FILE_FILEPATH)


@pytest.mark.parametrize("model_name", _EXPECTED_TRAINING_CFG_PARAMS.keys())
def test_training_def_file_params(tdf: TrainingConfiguration, model_name):
    """
    test that each model defined in the test json generate the
    expected params
    """
    params = tdf.get_params(model_name)
    del cast(dict, params)["algorithm"]
    assert params == _EXPECTED_TRAINING_CFG_PARAMS[model_name]


def test_training_def_file_model_names(tdf: TrainingConfiguration):
    """test that each model defined in the test json generate the
    expected params"""
    assert set(tdf.model_names) == set(_EXPECTED_TRAINING_CFG_PARAMS.keys())


def _finalize_expected_params(params: _TrainingParamsDict, cmdline_strs: list[str]):
    assert params["train_params"]

    n_jobs = (
        int(cmdline_strs[cmdline_strs.index("--tpot_jobs") + 1])
        if "--tpot_jobs" in cmdline_strs
        else params["train_params"]["n_jobs"]
    )
    max_eval_time_mins = (
        int(cmdline_strs[cmdline_strs.index("--max_eval_time_mins") + 1])
        if "--max_eval_time_mins" in cmdline_strs
        else params["train_params"]["max_eval_time_mins"]
    )
    max_time_mins = (
        int(cmdline_strs[cmdline_strs.index("--max_time_mins") + 1])
        if "--max_time_mins" in cmdline_strs
        else params["train_params"]["max_time_mins"]
    )
    generations = (
        int(cmdline_strs[cmdline_strs.index("--epochs_max") + 1])
        if "--epochs_max" in cmdline_strs
        else params["train_params"]["epochs_max"]
    )
    early_stop = (
        int(cmdline_strs[cmdline_strs.index("--early_stop") + 1])
        if "--early_stop" in cmdline_strs
        else params["train_params"]["early_stop"]
    )

    train_params = {
        # "use_dask": False,
        # "verbosity": 3,
        # "random_state": params["seed"],
        "max_time_mins": max_time_mins,
        "max_eval_time_mins": max_eval_time_mins,
        "n_jobs": n_jobs,
        "generations": generations,
        "early_stop": early_stop,
    }
    target_tuple = (
        params["target"].split(":") if isinstance(params["target"], str) else params["target"]
    )

    return train_params, target_tuple


@pytest.mark.parametrize(
    "cmdline, expected_reuse",
    [
        ("", False),
        ("--reuse", True),
        ("--algorithm tpot-light", False),
        ("--tpot_jobs 5", False),
        ("--max_time_mins 8 --max_eval_time_mins 4", False),
    ],
)
def test_training_def_file_train_test(
    mocker: MockFixture, cmdline: str, expected_reuse: bool, tdf: TrainingConfiguration
):
    """test that the calls to load_data and model_and_test are
    as expected"""
    cmdline_strs = cmdline.split(" ")
    model_name = "MLB-H-DK"
    expected_params = tdf.get_params(model_name)
    train_params, target_tuple = _finalize_expected_params(expected_params, cmdline_strs)

    algorithm = (
        cmdline_strs[cmdline_strs.index("--algorithm") + 1]
        if "--algorithm" in cmdline_strs
        else DEFAULT_ALGORITHM
    )

    fake_raw_df = mocker.Mock()
    fake_tt_data = mocker.Mock()
    mock_load_data = mocker.patch(
        "lib.pt_model.cfg.load_data", return_value=(fake_raw_df, fake_tt_data, None)
    )
    mock_model_and_test = mocker.patch("lib.pt_model.cfg.model_and_test", return_value=None)

    main(f"train {cmdline} {_TEST_DEF_FILE_FILEPATH} {model_name}")

    mock_load_data.assert_called_once_with(
        expected_params["data_filename"],
        target_tuple,
        expected_params["validation_season"],
        expected_params["seed"],
        include_position=expected_params["include_pos"],
        col_drop_filters=expected_params["cols_to_drop"],
        missing_data_threshold=expected_params["missing_data_threshold"],
        filtering_query=expected_params["filtering_query"],
        limit=None,
        expected_cols=None,
    )

    data_src_params = {k_: expected_params[k_] for k_ in _DATA_SRC_PARAMS.__args__}

    mock_model_and_test.assert_called_once_with(
        model_name,
        expected_params["validation_season"],
        fake_tt_data,
        target_tuple,
        algorithm,
        expected_params["p_or_t"],
        expected_params["recent_games"],
        expected_params["training_seasons"],
        train_params,
        expected_params["target_pos"],
        expected_params["training_pos"] or expected_params["target_pos"],
        ".",
        expected_reuse,
        model_dest_filename=None,
        data_src_params=data_src_params,
    )


def _create_expected_model_dict(
    model_name,
    feature_stat,
    feature_col,
    pos_col,
    dt,
    pkl_filepath,
    model_filepath,
    algorithm,
    expected_r2,
    expected_mae,
):
    expected_training_data_def = {
        k_: v_
        for k_, v_ in _EXPECTED_TRAINING_CFG_PARAMS[model_name].items()
        if k_
        not in [
            "seed",
            "validation_season",
            "data_filename",
            "filtering_query",
            "train_params",
            "cols_to_drop",
            "missing_data_threshold",
            "target_pos",
        ]
    }
    target = expected_training_data_def.pop("target")
    if isinstance(target, str):
        target = target.split(":")
        assert len(target) == 2
    features: dict[str, list | None] = {fname: None for fname in FeatureType.__args__}
    features["stat"] = [feature_stat]
    expected_training_data_def.update(
        {
            "features": features,
            "input_cols": [feature_col, pos_col],
            "seasons": expected_training_data_def.pop("training_seasons"),
            "recent_explode": True,
            "recent_mean": True,
            "target": [target[0], expected_training_data_def.pop("p_or_t").value, target[1]],
        }
    )
    artifact_parent_dir, artifact_filename = os.path.split(pkl_filepath)
    model_parent_dir = os.path.dirname(model_filepath)
    final_artifact_filepath = (
        pkl_filepath if model_parent_dir != artifact_parent_dir else artifact_filename
    )
    return {
        "name": model_name,
        "dt_trained": dt.isoformat(),
        "parameters": {
            "algorithm": algorithm,
            "filtering_query": None,
            "missing_data_threshold": _EXPECTED_TRAINING_CFG_PARAMS[model_name][
                "missing_data_threshold"
            ],
            "data_filename": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["data_filename"],
            **TRAINING_PARAM_DEFAULTS["dummy"][0],
        },
        "trained_parameters": {"regressor_path": final_artifact_filepath},
        "training_data_def": expected_training_data_def,
        "func_type": PTPredictModel.FUNC_TYPE_NAME,
        "meta_extra": {
            "performance": {
                "mae": expected_mae,
                "r2": expected_r2,
                "season": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["validation_season"],
            },
            "player_positions": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["target_pos"],
            "type": "sklearn",
            "trained_on_uname": platform.uname()._asdict(),
        },
    }


def _fake_metrics(mocker):
    """mock the performance calculations"""
    mock_sklearn = mocker.patch("lib.pt_model.train_test.sklearn")
    mock_sklearn.model_selection.train_test_split = sklearn.model_selection.train_test_split
    expected_r2 = round(random.random(), 3)
    expected_mae = round(random.random(), 3)
    mock_sklearn.metrics.r2_score.return_value = expected_r2
    mock_sklearn.metrics.mean_absolute_error.return_value = expected_mae
    return expected_r2, expected_mae


def test_model_gen(tmpdir, mocker):
    """test that the resulting model file is as expected and that
    the expected calls to fit the model, etc were made"""
    model_name = "MLB-H-DK"
    algorithm = "dummy"
    target_calc_stat = "dk_score"
    feature_stat = "hits"
    target_col = "calc:" + target_calc_stat
    position = "C"
    expected_r2, expected_mae = _fake_metrics(mocker)

    cmdline = (
        f"--algorithm {algorithm} --max_time_mins 8 --max_eval_time_mins 4 "
        f"--dest_dir {tmpdir} {_TEST_DEF_FILE_FILEPATH} {model_name}"
    )

    mock_pd = mocker.patch("lib.pt_model.train_test.pd")
    mock_pd.read_parquet.return_value = pd.DataFrame(
        {"pos": [position], "pos_id": [1], "extra:bases": [0], target_col: [0], "calc:y_score": [0]}
    )

    feature_col = f"stat:{feature_stat}:std"
    pos_col = f"pos_{position}"
    mock_pd.get_dummies.return_value = pd.DataFrame(
        {
            feature_col: [8, 6, 7, 5],
            pos_col: [1] * 4,
            "season": [2023] * 4,
            target_col: [1, 2, 3, 4],
        }
    )

    dt = datetime(2023, 12, 3, 0, 33)
    with freeze_time(dt):
        main("train " + cmdline)

    dest_filepath_base = os.path.join(
        tmpdir, f"{model_name}.{target_calc_stat}.{algorithm}.{dt_to_filename_str(dt)}"
    )

    pkl_filepath = dest_filepath_base + ".pkl"
    with open(pkl_filepath, "rb") as f_:
        regressor = joblib.load(f_)
    assert (
        isinstance(regressor, DummyRegressor)
        and regressor.strategy == TRAINING_PARAM_DEFAULTS["dummy"][0]["strategy"]
    )

    model_filepath = dest_filepath_base + ".model"
    with open(model_filepath, "r") as f_:
        model_dict = json.load(f_)

    del model_dict["model_file_version"]
    expected_model_dict = _create_expected_model_dict(
        model_name,
        feature_stat,
        feature_col,
        pos_col,
        dt,
        pkl_filepath,
        model_filepath,
        algorithm,
        expected_r2,
        expected_mae,
    )
    deep_compare_dicts(model_dict, expected_model_dict)


def _retrain_test_model_check(
    filepath_base: str,
    expected_r2,
    expected_mae,
    mock_pickle_func,
    mock_regressor_cls,
    mock_fit,
    artifact_ext,
    # TODO: this arg in general does not seem necessary, just get the params from the defaults dict
    expected_params: dict,
):
    """
    assert various things after training the original or retrained
    model during retrain testing

    returns: model dict without keys that should not match
    """
    mock_regressor_cls.assert_called_once()
    expected_model_artifact_path = filepath_base + "." + artifact_ext
    mock_pickle_func.assert_called_once_with(mock_fit, expected_model_artifact_path)
    with open(filepath_base + ".model", "r") as f_:
        model_dict = cast(dict, json.load(f_))

    assert model_dict["trained_parameters"]["regressor_path"].rsplit(".", 1)[0] == filepath_base
    del model_dict["trained_parameters"]["regressor_path"]

    # test and drop relevent parameters
    algo_defaults, renames = TRAINING_PARAM_DEFAULTS[expected_params["algorithm"]]
    algo_param_keys = set(algo_defaults.keys())
    algo_param_keys.add("algorithm")
    for param_key in algo_param_keys:
        if param_key not in expected_params:
            continue
        model_key = renames.get(param_key, param_key) if renames else param_key
        assert model_dict["parameters"][model_key] == expected_params[param_key]
        del model_dict["parameters"][model_key]

    assert model_dict["meta_extra"]["performance"]["r2"] == expected_r2
    assert model_dict["meta_extra"]["performance"]["mae"] == expected_mae
    for param_key in ["performance", "type"]:
        del model_dict["meta_extra"][param_key]

    for param_key in ["dt_trained"]:
        del model_dict[param_key]

    return model_dict


def _infer_expected_params(algo: AlgorithmType, tdf_params: dict | None, override_params: dict):
    param_default = TRAINING_PARAM_DEFAULTS[algo][0]
    params = {"algorithm": algo}

    for param_key, default_value in param_default.items():
        if param_key in override_params:
            value = override_params[param_key]
        elif tdf_params is not None and param_key in tdf_params["train_params"]:
            value = tdf_params["train_params"][param_key]
        elif default_value != _NO_DEFAULT:
            value = default_value
        else:
            continue
        params[param_key] = value
    return params


_UNSUPPORTED_BY_CLI = {"strategy", "batch_size", "hidden_layers"}
"""
model parameters not supported by regression cli, which should therefore not
by used to create command line args during retry testing
"""


def _train_prep(mocker, params, algo: AlgorithmType, tdf_params: dict | None):
    """
    prep for train/retrain during the retrain test

    params: all known/available training parameters, irrelevant parameters or\
        params that should not be on the command line be dropped
    """
    artifact_ext = "pkl"

    if algo == "nn":
        mock_pickle_func = mocker.patch("lib.pt_model.train_test.torch").save
        mock_regressor = mocker.patch("lib.pt_model.train_test.NNRegressor")
        mock_fitted = mock_regressor.return_value.to.return_value.fit.return_value
        artifact_ext = "pt"
    elif algo.startswith("tpot"):
        mock_pickle_func = mocker.patch("lib.pt_model.train_test.joblib").dump
        mock_regressor = mocker.patch("lib.pt_model.train_test.TPOTRegressor")
        mock_fitted = mock_regressor.return_value.fit.return_value.fitted_pipeline_
    elif algo == "dummy":
        mock_pickle_func = mocker.patch("lib.pt_model.train_test.joblib").dump
        mock_regressor = mocker.patch("lib.pt_model.train_test.DummyRegressor")
        mock_fitted = mock_regressor.return_value.fit.return_value
    else:
        raise NotImplementedError(f"{algo=} not supported")

    param_defaults = TRAINING_PARAM_DEFAULTS[algo][0]
    cli_params = {
        k_: params[k_]
        for k_ in param_defaults
        if params.get(k_) is not None and k_ not in _UNSUPPORTED_BY_CLI
    }
    cli_args = " ".join([f"--{key} {value}" for key, value in cli_params.items()])

    r2, mae = _fake_metrics(mocker)
    expected_train_params = _infer_expected_params(algo, tdf_params, params)

    return (
        cli_args,
        r2,
        mae,
        mock_fitted,
        mock_regressor,
        artifact_ext,
        mock_pickle_func,
        expected_train_params,
    )


@pytest.mark.parametrize(
    "retrain_algo, retrain_params",
    [
        ("dummy", {}),
        ("nn", {"early_stop_epochs": 10, "epochs_max": 1000}),
        ("tpot-light", {"n_jobs": 2}),
        ("tpot", {"max_time_mins": 120, "max_eval_time_mins": 15}),
    ],
    ids=["ReDummy", "ReNN", "ReTpotlight", "ReTpot"],
)
@pytest.mark.parametrize(
    "orig_algo, orig_params",
    [
        ("dummy", {}),
        ("nn", {"early_stop_epochs": 5, "epochs_max": 500, "checkpoint_dir": "/tmp/check"}),
        ("tpot-light", {"n_jobs": 4, "max_time_mins": 45}),
        ("tpot", {"n_jobs": 5, "epochs_max": 10, "early_stop": 2}),
    ],
    ids=["dummy", "nn", "tpotlight", "tpot"],
)
@pytest.mark.parametrize("retrain_w_tdf", [False, True], ids=["ReWO/Tdf", "ReW/Tdf"])
def test_retrain(
    mocker: MockFixture,
    tmpdir,
    tdf,
    orig_algo: AlgorithmType,
    orig_params: dict,
    retrain_algo: AlgorithmType,
    retrain_params: dict,
    retrain_w_tdf: bool,
):
    """
    Ensure that retraining a model leads to proper model training params by using
    the cli to train a model, then retrain using different command line args.
    Test that the resulting models have the expected parameters
    """
    model_name = "MLB-H-DK"
    tdf_params = tdf.get_params(model_name)

    # mock the training data load
    fake_raw_df = mocker.Mock(name="fake-raw-training-df")
    fake_training_features_df = pd.DataFrame({"stat:fake-stat:std": [8, 6, 7, 5], "pos_C": [1] * 4})
    fake_tt_data = [fake_training_features_df, None, fake_training_features_df, None, None, None]
    mocker.patch("lib.pt_model.cfg.load_data", return_value=(fake_raw_df, fake_tt_data, None))

    # mocks to skip artifact dumping stuff
    mock_tt_os = mocker.patch("lib.pt_model.train_test.os")
    mock_tt_os.path.join = os.path.join
    mock_tt_os.path.isfile.return_value = False
    mocker.patch.object(PTPredictModel, "_dump_artifacts")

    (
        cli_args,
        expected_r2_1,
        expected_mae_1,
        model_1_fitted,
        model_1_regressor,
        expected_artifact_ext,
        mock_pickle_func,
        expected_train_1_params,
    ) = _train_prep(mocker, orig_params, orig_algo, tdf_params)

    main(
        f"train --algorithm {orig_algo} {cli_args} --dest_filename model-1 --dest_dir "
        f"{tmpdir} {_TEST_DEF_FILE_FILEPATH} {model_name}"
    )

    model_1_dict = _retrain_test_model_check(
        model_1_filepath_base := os.path.join(tmpdir, "model-1"),
        expected_r2_1,
        expected_mae_1,
        mock_pickle_func,
        model_1_regressor,
        model_1_fitted,
        expected_artifact_ext,
        expected_train_1_params,
    )

    (
        cli_args,
        expected_r2_2,
        expected_mae_2,
        mock_model_2_fitted,
        model_2_regressor,
        expected_artifact_ext,
        mock_pickle_func,
        expected_train_2_params,
    ) = _train_prep(
        mocker,
        {**expected_train_1_params, **retrain_params},
        retrain_algo,
        tdf_params if retrain_w_tdf else None,
    )

    orig_cfg_args = f"--orig_cfg_file {_TEST_DEF_FILE_FILEPATH}" if retrain_w_tdf else ""

    main(
        f"retrain {model_1_filepath_base}.model --dest_filename model-2 "
        f"--algo {retrain_algo} --dest_dir {tmpdir} {orig_cfg_args} {cli_args}"
    )

    model_2_dict = _retrain_test_model_check(
        os.path.join(tmpdir, "model-2"),
        expected_r2_2,
        expected_mae_2,
        mock_pickle_func,
        model_2_regressor,
        mock_model_2_fitted,
        expected_artifact_ext,
        expected_train_2_params,
    )

    assert model_1_dict == model_2_dict
