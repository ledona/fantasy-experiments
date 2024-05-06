import json
import os
import platform
from datetime import datetime

import joblib
import pandas as pd
import pytest
from fantasy_py import FeatureType, PlayerOrTeam
from fantasy_py.inference import PTPredictModel
from freezegun import freeze_time
from ledona import deep_compare_dicts
from pytest_mock import MockFixture
from sklearn.dummy import DummyRegressor

from ..regressor import (
    _DEFAULT_ARCHITECTURE,
    _DUMMY_REGRESSOR_KWARGS,
    _TrainingDefinitionFile,
    _Params,
    main,
)
from ..train_test import dt_to_filename_str

_EXPECTED_PARAMS = {
    "win": {
        "target": ("stat", "win"),
        "seed": 1,
        "validation_season": 2021,
        "recent_games": 5,
        "training_seasons": [2022, 2023],
        "missing_data_threshold": 0.07,
        "data_filename": "all-data.pq",
        "train_params": {"max_time_mins": 10, "max_eval_time_mins": 2},
    },
    "team-pts": {
        "target": ("stat", "pts"),
        "seed": 1,
        "validation_season": 2021,
        "recent_games": 5,
        "training_seasons": [2022, 2023],
        "missing_data_threshold": 0.07,
        "data_filename": "team.pq",
        "p_or_t": PlayerOrTeam.TEAM,
        "train_params": {"max_time_mins": 10, "max_eval_time_mins": 2},
    },
    "team-pts-allowed": {
        "target": ("stat", "pts-allowed"),
        "seed": 1,
        "validation_season": 2021,
        "recent_games": 3,
        "training_seasons": [2022, 2023],
        "missing_data_threshold": 0.07,
        "data_filename": "team.pq",
        "p_or_t": PlayerOrTeam.TEAM,
        "train_params": {"max_time_mins": 10, "max_eval_time_mins": 2},
    },
    "player-score": {
        "target": ("stat", "score"),
        "seed": 1,
        "validation_season": 2021,
        "recent_games": 5,
        "training_seasons": [2022, 2023],
        "missing_data_threshold": 0.07,
        "data_filename": "player.pq",
        "include_pos": True,
        "p_or_t": PlayerOrTeam.PLAYER,
        "target_pos": ["P1", "P2"],
        "training_pos": ["P3", "P4", "P5"],
        "train_params": {"max_time_mins": 25, "max_eval_time_mins": 2},
    },
    "p1-stop": {
        "target": ("stat", "stop"),
        "seed": 1,
        "validation_season": 2021,
        "recent_games": 5,
        "training_seasons": [2022, 2023],
        "missing_data_threshold": 0.07,
        "data_filename": "player.pq",
        "include_pos": False,
        "p_or_t": PlayerOrTeam.PLAYER,
        "target_pos": ["P1"],
        "training_pos": ["P3", "P4", "P5"],
        "train_params": {"max_time_mins": 25, "max_eval_time_mins": 5},
    },
}

_DIRNAME = os.path.dirname(os.path.abspath(__file__))

for model_params in _EXPECTED_PARAMS.values():
    pad_dict = {
        param_key: None
        for param_key, value_type in _Params.__annotations__.items()
        if hasattr(value_type, "__args__")
        and type(None) in value_type.__args__
        and param_key not in model_params
    }
    model_params.update(pad_dict)
    #     {"data_filename": os.path.join(_DIRNAME, model_params["data_filename"]), **pad_dict}
    # )


_TEST_DEF_FILE_FILEPATH = os.path.join(_DIRNAME, "test.json")


@pytest.fixture(name="tdf", scope="module")
def _tdf():
    return _TrainingDefinitionFile(_TEST_DEF_FILE_FILEPATH)


@pytest.mark.parametrize("model_name", _EXPECTED_PARAMS.keys())
def test_training_def_file_params(tdf: _TrainingDefinitionFile, model_name):
    """test that each model defined in the test json generate the
    expected params"""
    params = tdf.get_params(model_name)
    assert params == _EXPECTED_PARAMS[model_name]


def test_training_def_file_model_names(tdf: _TrainingDefinitionFile):
    """test that each model defined in the test json generate the
    expected params"""
    assert set(tdf.model_names) == set(_EXPECTED_PARAMS.keys())


@pytest.mark.parametrize(
    "cmdline, expected_reuse",
    [
        ("", False),
        ("--reuse", True),
        ("--arch tpot-light", False),
        ("--tpot_jobs 5", False),
        ("--max_time_mins 8 --max_eval_time_mins 4", False),
    ],
)
def test_training_def_file_train_test(
    mocker: MockFixture, cmdline: str, expected_reuse: bool, tdf: _TrainingDefinitionFile
):
    """test that the calls to load_data and model_and_test are
    as expected"""
    cmdline_strs = cmdline.split(" ")
    model_name = "p1-stop"
    params = tdf.get_params(model_name)

    expected_tpot_train_params = {
        "use_dask": False,
        "verbosity": 3,
        "random_state": params["seed"],
        "max_time_mins": params["train_params"]["max_time_mins"],
        "max_eval_time_mins": params["train_params"]["max_eval_time_mins"],
    }

    expected_tpot_train_params["n_jobs"] = (
        int(cmdline_strs[cmdline_strs.index("--tpot_jobs") + 1])
        if "--tpot_jobs" in cmdline_strs
        else None
    )
    if "--max_eval_time_mins" in cmdline_strs:
        expected_tpot_train_params["max_eval_time_mins"] = int(
            cmdline_strs[cmdline_strs.index("--max_eval_time_mins") + 1]
        )
    if "--max_time_mins" in cmdline_strs:
        expected_tpot_train_params["max_time_mins"] = int(
            cmdline_strs[cmdline_strs.index("--max_time_mins") + 1]
        )

    arch = (
        cmdline_strs[cmdline_strs.index("--arch") + 1]
        if "--arch" in cmdline_strs
        else _DEFAULT_ARCHITECTURE
    )

    fake_raw_df = mocker.Mock()
    fake_tt_data = mocker.Mock()
    mock_load_data = mocker.patch(
        "lib.regressor.load_data", return_value=(fake_raw_df, fake_tt_data, None)
    )
    mock_model_and_test = mocker.patch("lib.regressor.model_and_test", return_value=None)

    main(f"train {cmdline} {_TEST_DEF_FILE_FILEPATH} {model_name}")

    mock_load_data.assert_called_once_with(
        params["data_filename"],
        params["target"],
        params["validation_season"],
        params["seed"],
        include_position=params["include_pos"],
        col_drop_filters=params["cols_to_drop"],
        missing_data_threshold=params["missing_data_threshold"],
        filtering_query=params["filtering_query"],
        limit=None,
    )

    mock_model_and_test.assert_called_once_with(
        model_name,
        params["validation_season"],
        fake_tt_data,
        params["target"],
        arch,
        params["p_or_t"],
        params["recent_games"],
        params["training_seasons"],
        expected_tpot_train_params,
        params["target_pos"],
        params["training_pos"] or params["target_pos"],
        ".",
        expected_reuse,
        model_dest_filename=None,
    )


def _create_expected_model_dict(
    model_name, feature_stat, feature_col, dt, pkl_filepath, model_filepath, algo_type
):
    expected_training_data_def = {
        k_: v_
        for k_, v_ in _EXPECTED_PARAMS[model_name].items()
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
    features: dict[str, list | None] = {fname: None for fname in FeatureType.__args__}
    features["stat"] = [feature_stat]
    expected_training_data_def.update(
        {
            "features": features,
            "input_cols": [feature_col],
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
        "parameters": {"algo_type": algo_type, **_DUMMY_REGRESSOR_KWARGS},
        "trained_parameters": {"regressor_path": final_artifact_filepath},
        "training_data_def": expected_training_data_def,
        "func_type": PTPredictModel.FUNC_TYPE_NAME,
        "meta_extra": {
            "performance": {
                "mae": 1.5,
                "r2": -9,
                "season": _EXPECTED_PARAMS[model_name]["validation_season"],
            },
            "player_positions": _EXPECTED_PARAMS[model_name]["target_pos"],
            "type": "sklearn",
            "trained_on_uname": platform.uname()._asdict(),
        },
    }


def test_model_gen(tmpdir, mocker):
    """test that the resulting model file is as expected and that
    the expected calls to fit the model, etc were made"""
    model_name = "p1-stop"
    arch = "dummy"
    cmdline = (
        f"--arch {arch} --max_time_mins 8 --max_eval_time_mins 4 "
        f"--dest_dir {tmpdir} {_TEST_DEF_FILE_FILEPATH} {model_name}"
    )

    mock_pd = mocker.patch("lib.train_test.pd")
    mock_pd.read_parquet.return_value = pd.DataFrame({"pos": [], "pos_id": []})
    target_stat = _EXPECTED_PARAMS[model_name]["target"][1]
    feature_stat = "start"
    feature_col = f"stat:{feature_stat}:std"
    mock_pd.get_dummies.return_value = pd.DataFrame(
        {
            "season": [2021, 2021, 2020, 2020],
            "stat:" + target_stat: [5, 4, 2, 3],
            feature_col: [8, 6, 7, 5],
        }
    )
    mocker.patch("lib.train_test.CLSRegistry")

    dt = datetime(2023, 12, 3, 0, 33)
    with freeze_time(dt):
        main("train " + cmdline)

    model_filepath = os.path.join(
        tmpdir, f"{model_name}.{target_stat}.{arch}.{dt_to_filename_str(dt)}.model"
    )
    pkl_filepath = os.path.join(
        tmpdir, f"{model_name}.{target_stat}.{arch}.{dt_to_filename_str(dt)}.pkl"
    )
    with open(pkl_filepath, "rb") as f_:
        regressor = joblib.load(f_)

    assert (
        isinstance(regressor, DummyRegressor)
        and regressor.strategy == _DUMMY_REGRESSOR_KWARGS["strategy"]
    )

    with open(model_filepath, "r") as f_:
        model_dict = json.load(f_)

    del model_dict["model_file_version"]
    expected_model_dict = _create_expected_model_dict(
        model_name, feature_stat, feature_col, dt, pkl_filepath, model_filepath, arch
    )
    deep_compare_dicts(model_dict, expected_model_dict)
