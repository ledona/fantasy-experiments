import os

import pytest
from pytest_mock import MockFixture

from fantasy_py import PlayerOrTeam

from ..cli import _DEFAULT_TPOT_JOBS, TrainingDefinitionFile, _Params, main, _DEFAULT_AUTOML_TYPE

_EXPECTED_PARAMS = {
    "win": {
        "target": ("stat", "win"),
        "seed": 1,
        "validation_season": 2021,
        "recent_games": 5,
        "training_seasons": [2022, 2023],
        "missing_data_threshold": 0.07,
        "data_filename": "all-data.pq",
        "train_params": {"max_train_mins": 10, "max_iter_mins": 2},
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
        "train_params": {"max_train_mins": 10, "max_iter_mins": 2},
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
        "train_params": {"max_train_mins": 10, "max_iter_mins": 2},
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
        "train_params": {"max_train_mins": 25, "max_iter_mins": 2},
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
        "train_params": {"max_train_mins": 25, "max_iter_mins": 5},
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
    model_params.update(
        {"data_filename": os.path.join(_DIRNAME, model_params["data_filename"]), **pad_dict}
    )


_TEST_DEF_FILE_FILEPATH = os.path.join(_DIRNAME, "test.json")


@pytest.fixture(name="tdf", scope="module")
def _tdf():
    return TrainingDefinitionFile(_TEST_DEF_FILE_FILEPATH)


@pytest.mark.parametrize("model_name", _EXPECTED_PARAMS.keys())
def test_training_def_file_params(tdf: TrainingDefinitionFile, model_name):
    """test that each model defined in the test json generate the
    expected params"""
    params = tdf.get_params(model_name)
    assert params == _EXPECTED_PARAMS[model_name]


def test_training_def_file_model_names(tdf: TrainingDefinitionFile):
    """test that each model defined in the test json generate the
    expected params"""
    assert set(tdf.model_names) == set(_EXPECTED_PARAMS.keys())


@pytest.mark.parametrize(
    "cmdline",
    [
        "",
        "--overwrite --reuse_existing",
        "--automl_type tpot-light",
        "--tpot_jobs 5",
        "--max_train_mins 8 --max_iter_mins 4",
    ],
)
def test_training_def_file_train_test(
    mocker: MockFixture, cmdline: str, tdf: TrainingDefinitionFile
):
    """test that the calls to load_data and model_and_test are
    as expected"""
    cmdline_strs = cmdline.split(" ")
    overwrite = "--overwrite" in cmdline
    reuse_existing = "--reuse_existing" in cmdline

    model_name = "p1-stop"
    params = tdf.get_params(model_name)

    expected_train_params = {
        "seed": params["seed"],
        "tpot_jobs": _DEFAULT_TPOT_JOBS,
        "max_iter_mins": None,
        "max_train_mins": None,
    }

    if "--tpot_jobs" in cmdline_strs:
        expected_train_params["tpot_jobs"] = int(
            cmdline_strs[cmdline_strs.index("--tpot_jobs") + 1]
        )
    if "--max_iter_mins" in cmdline_strs:
        expected_train_params["max_iter_mins"] = int(
            cmdline_strs[cmdline_strs.index("--max_iter_mins") + 1]
        )
    if "--max_train_mins" in cmdline_strs:
        expected_train_params["max_train_mins"] = int(
            cmdline_strs[cmdline_strs.index("--max_train_mins") + 1]
        )
    if "--automl_type" in cmdline_strs:
        automl_type = cmdline_strs[cmdline_strs.index("--automl_type") + 1]
    else:
        automl_type = _DEFAULT_AUTOML_TYPE

    fake_raw_df = mocker.Mock()
    fake_tt_data = mocker.Mock()
    mock_load_data = mocker.patch(
        "lib.cli.load_data", return_value=(fake_raw_df, fake_tt_data, None)
    )
    mock_model_and_test = mocker.patch("lib.cli.model_and_test", return_value=None)

    main(cmdline + f" {_TEST_DEF_FILE_FILEPATH} {model_name}")

    mock_load_data.assert_called_once_with(
        params["data_filename"],
        params["target"],
        params["validation_season"],
        include_position=params["include_pos"],
        col_drop_filters=params["cols_to_drop"],
        seed=params["seed"],
        missing_data_threshold=params["missing_data_threshold"],
        filtering_query=params["filtering_query"],
    )

    mock_model_and_test.assert_called_once_with(
        model_name,
        params["validation_season"],
        fake_tt_data,
        params["target"],
        automl_type,
        params["p_or_t"],
        params["recent_games"],
        params["training_seasons"],
        expected_train_params,
        params["target_pos"],
        params["training_pos"] or params["target_pos"],
        raw_df=fake_raw_df,
        reuse_existing=reuse_existing,
        overwrite=overwrite,
        dest_dir=None,
    )


def test_model_gen(tmpdir, mocker):
    """test that the resulting model file is as expected and that
    the expected calls to fit the model, etc were made"""
    model_name = "p1-stop"
    cmdline = (
        f"--automl_type dummy --tpot_jobs 5 --max_train_mins 8 --max_iter_mins 4 "
        f"--dest_dir {tmpdir} {_TEST_DEF_FILE_FILEPATH} {model_name}"
    )

    mock_pd = mocker.patch("lib.train_test.pd")

    main(cmdline)
    raise NotImplementedError()
