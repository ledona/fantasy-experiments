import json
import os
import platform
import random
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
import sklearn
import torch
from autogluon.tabular import TabularPredictor
from fantasy_py import FeatureType, PlayerOrTeam, dt_to_filename_str
from fantasy_py.inference import NNRegressor, PTPredictModel
from freezegun import freeze_time
from ledona import deep_compare_dicts
from pytest_mock import MockFixture
from sklearn.dummy import DummyRegressor
from tpot2 import TPOTRegressor

from ..pt_model import (
    TRAINING_PARAM_DEFAULTS,
    AlgorithmType,
    TrainingConfiguration,
    _TrainingParamsDict,
)
from ..pt_model.cfg import _NO_DEFAULT
from ..regressor import main

_VALIDATION_SEASON = 2023
"""validation season, must match the model training definition"""

_EXPECTED_TRAINING_CFG_PARAMS = {
    "MLB-team-runs": {
        "sport": "mlb",
        "seed": 1,
        "p_or_t": PlayerOrTeam.TEAM,
        "cols_to_drop": [
            ".*:p.*",
            ".*opp-team",
            "extra:(whip|venue|opp|hit|is_home).*",
            "stat:.*:.*",
        ],
        "missing_data_threshold": 0.1,
        "train_params": {
            "epochs_max": 100,
            "early_stop": 5,
            "max_eval_time_mins": 15,
            "max_time_mins": 60,
        },
        "validation_season": _VALIDATION_SEASON,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_team.parquet",
        "target": "stat:off_runs",
    },
    "MLB-P-DK": {
        "sport": "mlb",
        "seed": 1,
        "p_or_t": PlayerOrTeam.PLAYER,
        "cols_to_drop": [".*dk_score:.*", ".*recent-.*", ".*y_score.*", "extra:venue.*"],
        "missing_data_threshold": 0.28,
        "train_params": {
            "epochs_max": 100,
            "early_stop": 5,
            "max_eval_time_mins": 15,
            "max_time_mins": 60,
        },
        "include_pos": False,
        "validation_season": _VALIDATION_SEASON,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_pitcher.parquet",
        "target": "calc:dk_score",
        "target_pos": ["P"],
    },
    "MLB-H-DK": {
        "sport": "mlb",
        "seed": 1,
        "p_or_t": PlayerOrTeam.PLAYER,
        "cols_to_drop": [".*y_score.*", "extra:bases"],
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
        "validation_season": _VALIDATION_SEASON,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_hitter.parquet",
        "target": "calc:dk_score",
        "include_pos": True,
        "target_pos": ["1B", "2B", "3B", "SS", "C", "LF", "RF", "CF", "OF"],
    },
    "MLB-H-hit": {
        "sport": "mlb",
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
        "validation_season": _VALIDATION_SEASON,
        "recent_games": 5,
        "training_seasons": [2021, 2022],
        "data_filename": "mlb_hitter.parquet",
        "target": "stat:off_hit",
        "include_pos": True,
        "target_pos": ["1B", "2B", "3B", "SS", "C", "LF", "RF", "CF", "OF"],
    },
}

_DIRNAME = os.path.dirname(os.path.abspath(__file__))

# add keys for Noneable parameters
for _model_params in _EXPECTED_TRAINING_CFG_PARAMS.values():
    if _model_params.get("training_pos") is None and _model_params.get("target_pos") is not None:
        _model_params["training_pos"] = _model_params.get("target_pos")
    pad_dict = {
        param_key: None
        for param_key, value_type in _TrainingParamsDict.__annotations__.items()
        if hasattr(value_type, "__args__")
        and type(None) in value_type.__args__
        and param_key not in _model_params
        and param_key != "original_model_columns"
    }
    _model_params.update(pad_dict)


_TEST_DEF_FILE_FILEPATH = os.path.join(_DIRNAME, "test.json")


@pytest.fixture(name="test_definition_file", scope="module")
def _test_definition_file():
    return TrainingConfiguration(_TEST_DEF_FILE_FILEPATH)


@pytest.mark.parametrize("model_name", _EXPECTED_TRAINING_CFG_PARAMS.keys())
def test_training_def_file_params(test_definition_file: TrainingConfiguration, model_name):
    """
    test that each model defined in the test json generate the
    expected params
    """
    params = test_definition_file.get_params(model_name)
    del cast(dict, params)["algorithm"]
    assert params == _EXPECTED_TRAINING_CFG_PARAMS[model_name]


class TestParamCascade:
    """test model parameter retrieval"""

    _ALGO = "algo"
    """name of algorithm used by algo specific params test"""

    _TEST_CASES = {
        "only-globals": ({"a": 1, "b": 2}, {}, {}, False, {"a": 1, "b": 2}),
        "only-globals-w-algo": ({"a": 1, "b": 2}, {}, {}, False, {"a": 1, "b": 2}),
        "only-group": ({}, {"a": 1, "b": 2}, {}, False, {"a": 1, "b": 2}),
        "only-model": ({}, {}, {"a": 1, "b": 2}, False, {"a": 1, "b": 2}),
        "all-levels-disjoint": ({"a": 1}, {"b": 2}, {"c": 3}, False, {"a": 1, "b": 2, "c": 3}),
        "mixed": ({"a": 1, "b": 2}, {"a": 3, "c": 5}, {"b": 6}, False, {"a": 3, "b": 6, "c": 5}),
        "ignore-algo-1": ({_ALGO + ".a": 1}, {"b": 2}, {"c": 3}, False, {"b": 2, "c": 3}),
        "ignore-algo-2": (
            {"a": 1, "b": 2},
            {"a": 3, "c": 5},
            {_ALGO + ".b": 6},
            False,
            {"a": 3, "b": 2, "c": 5},
        ),
        "ignore-algo-3": (
            {"a": 1.1, _ALGO + ".a": 1, "xx.a": 2},
            {"b": 3, _ALGO + ".a": 5},
            {_ALGO + ".b": 6},
            False,
            {"a": 1.1, "b": 3},
        ),
        "w-algo-1": (
            {"a": 1, "b": 2},
            {"a": 3, "c": 5},
            {_ALGO + ".b": 6},
            True,
            {"a": 3, "b": 6, "c": 5},
        ),
        "w-algo-2": (
            {"a": 1, _ALGO + ".b": 2},
            {"a": 3, "c": 5},
            {"b": 6},
            True,
            {"a": 3, "b": 2, "c": 5},
        ),
        "w-algo-3": (
            {_ALGO + ".a": 1, _ALGO + ".b": 2},
            {"a": 3, "c": 5},
            {"b": 6},
            True,
            {"a": 1, "b": 2, "c": 5},
        ),
        "w-algo-use-and-ignore": (
            {_ALGO + ".a": 1, _ALGO + ".b": 2, "c": 10},
            {"a": 3, "XX.c": 5},
            {"b": 6},
            True,
            {"a": 1, "b": 2, "c": 10},
        ),
        "w-algo-complex": (
            {_ALGO + ".a": 1, _ALGO + ".b": 2, "c": 10},
            {"a": 3, "XX.c": 5},
            {_ALGO + ".b": 6},
            True,
            {"a": 1, "b": 6, "c": 10},
        ),
        "w-algo-multi": (
            {"a": 1.1, _ALGO + ".a": 1, "xx.a": 2},
            {"b": 3, "XX.c": 5},
            {_ALGO + ".b": 6},
            True,
            {"a": 1, "b": 6},
        ),
        "w-algo-multi-complex": (
            {"a": 1.1, _ALGO + ".a": 1, "xx.a": 2},
            {"b": 3, _ALGO + ".a": 5},
            {_ALGO + ".b": 6},
            True,
            {"a": 5, "b": 6},
        ),
    }

    @pytest.mark.parametrize(
        "global_params, group_params, model_params, use_algo, expected_params",
        list(_TEST_CASES.values()),
        ids=list(_TEST_CASES.keys()),
    )
    def test(
        self,
        mocker,
        global_params: dict,
        group_params: dict,
        model_params: dict,
        use_algo: bool,
        expected_params: dict,
    ):
        """test that training params are cascaded correctly"""
        mocker.patch.dict(
            TRAINING_PARAM_DEFAULTS,
            {self._ALGO: {"a": _NO_DEFAULT, "b": _NO_DEFAULT, "c": _NO_DEFAULT}},
        )
        final_params = TrainingConfiguration._params_from_cfg_levels(
            (self._ALGO if use_algo else "no-algo"),
            {"train_params": global_params},
            {"train_params": group_params},
            {"train_params": model_params},
        )[0]
        assert final_params == expected_params


def test_training_def_file_model_names(test_definition_file: TrainingConfiguration):
    """test that each model defined in the test json generate the
    expected params"""
    assert set(test_definition_file.model_names) == set(_EXPECTED_TRAINING_CFG_PARAMS.keys())


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
    limit: int | None,
):
    expected_training_data_def = {
        k_: v_
        for k_, v_ in _EXPECTED_TRAINING_CFG_PARAMS[model_name].items()
        if k_
        not in [
            "sport",
            "seed",
            "validation_season",
            "data_filename",
            "filtering_query",
            "train_params",
            "cols_to_drop",
            "missing_data_threshold",
            "target_pos",
            "limit",
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
            "input_cols": [pos_col, feature_col],
            "seasons": expected_training_data_def.pop("training_seasons"),
            "recent_explode": True,
            "recent_mean": True,
            "target": [target[0], expected_training_data_def.pop("p_or_t").value, target[1]],
        }
    )
    if limit is not None:
        expected_training_data_def["limit"] = limit

    artifact_parent_dir, artifact_filename = os.path.split(pkl_filepath)
    model_parent_dir = os.path.dirname(model_filepath)
    final_artifact_filepath = (
        pkl_filepath if model_parent_dir != artifact_parent_dir else artifact_filename
    )

    return {
        "name": model_name,
        "sport": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["sport"],
        "dt_trained": dt.isoformat(),
        "parameters": {
            "algorithm": algorithm,
            "filtering_query": None,
            "missing_data_threshold": _EXPECTED_TRAINING_CFG_PARAMS[model_name][
                "missing_data_threshold"
            ],
            "data_filename": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["data_filename"],
            **TRAINING_PARAM_DEFAULTS["dummy"],
        },
        "trained_parameters": {"regressor_path": final_artifact_filepath},
        "training_data_def": expected_training_data_def,
        "func_type": PTPredictModel.FUNC_TYPE_NAME,
        "meta_extra": {
            "performance": {
                "mae_val": expected_mae,
                "r2_val": expected_r2,
                "mae_test": expected_mae,
                "r2_test": expected_r2,
                "mae_train": expected_mae,
                "r2_train": expected_r2,
                "season_val": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["validation_season"],
            },
            "player_positions": _EXPECTED_TRAINING_CFG_PARAMS[model_name]["target_pos"],
            "type": "sklearn",
            "trained_on_uname": platform.uname()._asdict(),
            "desc_info": {
                "time_to_fit": str(timedelta()),
                "n_train_cases": 3,
                "n_test_cases": 1,
                "n_validation_cases": 4,
            },
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


@pytest.mark.parametrize("limit", [None, 10000], ids=["no-limit", "w-limit"])
def test_model_gen(tmpdir, mocker: MockFixture, limit: int | None):
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
    loaded_data_df = pd.DataFrame(
        {
            "pos": [position],
            "pos_id": [1],
            "extra:bases": [0],
            target_col: [0],
            "calc:y_score": [0],
            "season": [_VALIDATION_SEASON],
        }
    )

    if limit is not None:
        cmdline += f" --limit {limit}"
        mock_pq = mocker.patch("lib.pt_model.train_test.pq")
        mock_pq.ParquetFile.return_value.iter_batches.return_value = iter([0])
        mock_pa = mocker.patch("lib.pt_model.train_test.pa")
        mock_pa.Table.from_batches.return_value.to_pandas.return_value = loaded_data_df
    else:
        mock_pd.read_parquet.return_value = loaded_data_df

    feature_col = f"stat:{feature_stat}:std"
    pos_col = f"pos_{position}"
    mock_pd.get_dummies.return_value = pd.DataFrame(
        {
            feature_col: [8, 6, 7, 5],
            pos_col: [1] * 4,
            "season": [_VALIDATION_SEASON] * 4,
            target_col: [1, 2, 3, 4],
        }
    )

    dt = datetime(_VALIDATION_SEASON, 12, 3, 0, 33, tzinfo=UTC)
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
        and regressor.strategy == TRAINING_PARAM_DEFAULTS["dummy"]["dmy:strategy"]
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
        limit,
    )
    deep_compare_dicts(model_dict, expected_model_dict)


def _check_model_params(
    filepath_base: str,
    expected_r2,
    expected_mae,
    mock_save_func,
    mock_regressor_cls,
    expected_params: dict,
    label: str,
):
    """
    assert various things after training the original or retrained
    model during retrain testing

    returns: model dict
    """
    mock_regressor_cls.assert_called_once()
    mock_save_func.assert_called_once()
    with open(filepath_base + ".model", "r") as f_:
        model_dict = cast(dict, json.load(f_))

    assert model_dict["trained_parameters"]["regressor_path"].rsplit(".", 1)[0] == filepath_base, (
        f"{label}: expected regressor path"
    )
    assert model_dict["meta_extra"]["performance"]["r2_val"] == expected_r2, f"{label}: expected_r2"
    assert model_dict["meta_extra"]["performance"]["mae_val"] == expected_mae, (
        f"{label}: expected_mae"
    )

    for param, expected_value in expected_params.items():
        assert model_dict["parameters"][param] == expected_value, (
            f"{label}: value at the model's request param '{param}' should match the expected value."
        )

    return model_dict


def _infer_expected_params(
    algo: AlgorithmType, test_definition_file_params: dict | None, override_params: dict
):
    param_default = TRAINING_PARAM_DEFAULTS[algo]
    params = {"algorithm": algo}

    for param_key, default_value in param_default.items():
        if param_key in override_params:
            value = override_params[param_key]
        elif (
            test_definition_file_params is not None
            and param_key in test_definition_file_params["train_params"]
        ):
            value = test_definition_file_params["train_params"][param_key]
        elif default_value != _NO_DEFAULT:
            value = default_value
        else:
            continue
        params[param_key] = value
    return params


def _train_prep(
    mocker: MockFixture,
    train_params,
    algo: AlgorithmType,
    tdf_params: dict | None,
    prev_algo: AlgorithmType | None = None,
    prev_regressor: Mock | None = None,
    prev_save_func: Mock | None = None,
):
    """
    prep for train/retrain during the retrain test

    tdf_params: training definition file params
    prev_algo: defined if a previous algorithm was already prepped. e.g. when retraining a model this will be set to
        the original model's algorithm. If the same as the previous then the other prevs will be reset and returned
    """
    if prev_algo:
        assert prev_regressor and prev_save_func
        mock_regressor = prev_regressor
        mock_regressor.reset_mock()
        mock_save_func = prev_save_func
        mock_save_func.reset_mock()

    if algo == "nn":
        if prev_algo != "nn":
            mock_save_func = mocker.MagicMock(name="fake-torch.save", autospec=True)
            mocker.patch("lib.pt_model.nn.torch.save", mock_save_func)
            mock_regressor = mocker.patch("lib.pt_model.nn.NNRegressor", autospec=True)
            mock_fitted = mock_regressor.return_value.to.return_value.fit.return_value
            mock_fitted.epochs_trained = 5
    elif algo == "autogluon":
        if prev_algo != "autogluon":
            mock_regressor = mocker.patch(
                "lib.pt_model.autogluon.TabularPredictor", spec=TabularPredictor
            )
            mock_regressor.return_value.path = "path-to-autogluon-artifacts"
            mock_save_func = mock_regressor.return_value.clone_for_deployment
            mock_regressor.return_value.info.return_value = {"version": "x.y.z"}
            mock_regressor.return_value.model_info.return_value = {"info": "all-da-info"}
    elif algo.startswith("tpot"):
        if prev_algo != "tpot":
            mock_save_func = mocker.patch("lib.pt_model.tpot.joblib").dump
            mock_regressor = mocker.patch("lib.pt_model.tpot.TPOTRegressor", autospec=True)
            mock_regressor.return_value.evaluated_individuals = mocker.MagicMock(
                name="fake-evaluated-individuals"
            )
            mock_regressor.return_value.evaluated_individuals.Generation.max.return_value = 1
            mock_regressor.return_value.predict = mocker.MagicMock(name="fake-predict")
            mock_regressor.return_value.fitted_pipeline_ = mocker.Mock(name="fake-pipeline")
    elif algo == "dummy":
        if prev_algo != "dummy":
            mock_save_func = mocker.patch("lib.pt_model.train_test.joblib").dump
            mock_regressor = mocker.patch("lib.pt_model.train_test.DummyRegressor", autospec=True)
            mock_fitted = mock_regressor.return_value.fit.return_value
    else:
        raise NotImplementedError(f"{algo=} not supported")

    param_defaults = TRAINING_PARAM_DEFAULTS[algo]
    cli_params = {k_: train_params[k_] for k_ in param_defaults if train_params.get(k_) is not None}
    cli_args = " ".join([f"--{key} {value}" for key, value in cli_params.items()])

    r2, mae = _fake_metrics(mocker)
    expected_train_params = _infer_expected_params(algo, tdf_params, train_params)

    return (
        cli_args,
        r2,
        mae,
        mock_regressor,
        mock_save_func,
        expected_train_params,
    )


@pytest.mark.parametrize(
    "retrain_w_def_file",
    [False, True],
    ids=["wo/tdf", "w/tdf"],
)
@pytest.mark.parametrize(
    "retrain_algo, retrain_params",
    [
        ("dummy", {"dmy:strategy": "mean"}),
        ("nn", {"early_stop": 10, "epochs_max": 1000}),
        # TODO: add back after tpot upgrade
        # ("tpot-light", {"n_jobs": 2}),
        ("tpot", {"max_time_mins": 120, "tp:max_eval_time_mins": 15}),
        ("autogluon", {"ag:preset": "high", "max_time_mins": 120}),
    ],
    ids=[
        ">dummy",
        ">nn",
        # TODO: add back after tpot upgrade
        # ">tpot-light",
        ">tpot",
        ">autogluon",
    ],
)
@pytest.mark.parametrize(
    "orig_algo, orig_params",
    [
        ("dummy", {"dmy:strategy": "quantile"}),
        (
            "nn",
            {
                "early_stop": 5,
                "epochs_max": 500,
                "nn:learning_rate": 0.9,
                "nn:checkpoint_dir": "/tmp/check",
            },
        ),
        # TODO: add back after tpot upgrade
        # ("tpot-light", {"n_jobs": 4, "max_time_mins": 45}),
        ("tpot", {"n_jobs": 5, "epochs_max": 10, "early_stop": 2, "tp:population_size": 100}),
        ("autogluon", {"ag:preset": "medium"}),
    ],
    ids=[
        "dummy>",
        "nn>",
        # TODO: add back after tpot upgrade
        # "tpotlight>",
        "tpot>",
        "autogluon>",
    ],
)
def test_train_retrain_params(
    mocker: MockFixture,
    tmpdir,
    test_definition_file,
    orig_algo: AlgorithmType,
    orig_params: dict,
    retrain_algo: AlgorithmType,
    retrain_params: dict,
    retrain_w_def_file: bool,
):
    """
    Ensure that when training/retraining models on cli expected params are used.
    Test execution:
    1) train model on cli
    2) test that the expected params are used
    3) retrain using different command line args
    4) Test that the resulting models have the expected parameters

    orig_params: original cli params
    retrain_params: cli params used during retrain
    retrain_w_def_file: use a training definition file
    """
    # make sure the test is valid
    assert (
        len(
            invalid_params := set(orig_params.keys())
            - set(TRAINING_PARAM_DEFAULTS[orig_algo].keys())
        )
        == 0
    ), f"original algo params should be subset of defaults: {invalid_params=}"
    assert (
        len(
            invalid_params := set(retrain_params.keys())
            - set(TRAINING_PARAM_DEFAULTS[retrain_algo].keys())
        )
        == 0
    ), f"retrain algo params should be subset of defaults: {invalid_params=}"

    model_name = "MLB-H-DK"
    tdf_params = test_definition_file.get_params(model_name)

    # mock the training data load
    fake_raw_df = mocker.Mock(name="fake-raw-training-df")
    fake_training_features_df = pd.DataFrame({"pos_C": [1] * 4, "stat:fake-stat:std": [8, 6, 7, 5]})
    fake_tt_data = [fake_training_features_df, None, fake_training_features_df, None, [0], None]
    mocker.patch("lib.pt_model.cfg.load_data", return_value=(fake_raw_df, fake_tt_data, None))

    # mocks to skip artifact dumping stuff
    mock_tt_os = mocker.patch("lib.pt_model.train_test.os")
    mock_tt_os.path.join = os.path.join
    mock_tt_os.path.isfile.return_value = False
    mocker.patch.object(PTPredictModel, "dump_artifacts")

    (
        cli_args,
        expected_orig_r2,
        expected_orig_mae,
        orig_model_regressor,
        mock_save_func,
        expected_orig_params,
    ) = _train_prep(mocker, orig_params, orig_algo, tdf_params)

    main(
        f"train --algorithm {orig_algo} {cli_args} --dest_filename orig-model --dest_dir "
        f"{tmpdir} {_TEST_DEF_FILE_FILEPATH} {model_name}"
    )
    orig_model_filepath_base = os.path.join(tmpdir, "orig-model")

    _check_model_params(
        orig_model_filepath_base,
        expected_orig_r2,
        expected_orig_mae,
        mock_save_func,
        orig_model_regressor,
        expected_orig_params,
        "orig-model",
    )

    (
        cli_args,
        expected_retrained_r2,
        expected_retrained_mae,
        retrained_model_regressor,
        mock_save_func,
        expected_retrain_params,
    ) = _train_prep(
        mocker,
        {**expected_orig_params, **retrain_params},
        retrain_algo,
        tdf_params if retrain_w_def_file else None,
        prev_algo=orig_algo,
        prev_regressor=orig_model_regressor,
        prev_save_func=mock_save_func,
    )

    orig_cfg_args = f"--orig_cfg_file {_TEST_DEF_FILE_FILEPATH}" if retrain_w_def_file else ""

    main(
        f"retrain {orig_model_filepath_base}.model --dest_filename model-2 "
        f"--algo {retrain_algo} --dest_dir {tmpdir} {orig_cfg_args} {cli_args}"
    )

    _check_model_params(
        os.path.join(tmpdir, "model-2"),
        expected_retrained_r2,
        expected_retrained_mae,
        mock_save_func,
        retrained_model_regressor,
        expected_retrain_params,
        "retrained-model",
    )
