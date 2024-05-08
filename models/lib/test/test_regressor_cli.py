import json
import os
import platform
from datetime import datetime

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

from ..pt_model import TrainingConfiguration, _TrainingParamsDict
from ..regressor import _DEFAULT_ARCHITECTURE, _DUMMY_REGRESSOR_KWARGS, main

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
    assert params == _EXPECTED_TRAINING_CFG_PARAMS[model_name]


def test_training_def_file_model_names(tdf: TrainingConfiguration):
    """test that each model defined in the test json generate the
    expected params"""
    assert set(tdf.model_names) == set(_EXPECTED_TRAINING_CFG_PARAMS.keys())


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
    mocker: MockFixture, cmdline: str, expected_reuse: bool, tdf: TrainingConfiguration
):
    """test that the calls to load_data and model_and_test are
    as expected"""
    cmdline_strs = cmdline.split(" ")
    model_name = "MLB-H-DK"
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
        "lib.pt_model.cfg.load_data", return_value=(fake_raw_df, fake_tt_data, None)
    )
    mock_model_and_test = mocker.patch("lib.pt_model.cfg.model_and_test", return_value=None)

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
    model_name,
    feature_stat,
    feature_col,
    pos_col,
    dt,
    pkl_filepath,
    model_filepath,
    algo_type,
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
            "algo_type": algo_type,
            "filtering_query": None,
            "missing_data_threshold": _EXPECTED_TRAINING_CFG_PARAMS[model_name][
                "missing_data_threshold"
            ],
            **_DUMMY_REGRESSOR_KWARGS,
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


def test_model_gen(tmpdir, mocker):
    """test that the resulting model file is as expected and that
    the expected calls to fit the model, etc were made"""
    model_name = "MLB-H-DK"
    arch = "dummy"
    target_calc_stat = "dk_score"
    feature_stat = "hits"
    target_col = "calc:" + target_calc_stat
    position = "C"

    cmdline = (
        f"--arch {arch} --max_time_mins 8 --max_eval_time_mins 4 "
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

    mock_sklearn = mocker.patch("lib.pt_model.train_test.sklearn")
    mock_sklearn.model_selection.train_test_split = sklearn.model_selection.train_test_split
    expected_r2 = 0.453
    expected_mae = 7.379
    mock_sklearn.metrics.r2_score.return_value = expected_r2
    mock_sklearn.metrics.mean_absolute_error.return_value = expected_mae

    dt = datetime(2023, 12, 3, 0, 33)
    with freeze_time(dt):
        main("train " + cmdline)

    dest_filepath_base = os.path.join(
        tmpdir, f"{model_name}.{target_calc_stat}.{arch}.{dt_to_filename_str(dt)}"
    )
    model_filepath = dest_filepath_base + ".model"
    pkl_filepath = dest_filepath_base + ".pkl"
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
        model_name,
        feature_stat,
        feature_col,
        pos_col,
        dt,
        pkl_filepath,
        model_filepath,
        arch,
        expected_r2,
        expected_mae,
    )
    deep_compare_dicts(model_dict, expected_model_dict)
