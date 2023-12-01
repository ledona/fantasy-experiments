import os

import pytest
from fantasy_py import PlayerOrTeam

from ..cli import TrainingDefinitionFile, _Params, main

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


@pytest.mark.parametrize("cmdline", [""])
def test_training_def_file_train_test(cmdline: str):
    """test that the calls to load_data and model_and_test are
    as expected"""
    main(cmdline + f" {_TEST_DEF_FILE_FILEPATH} p1-stop")
    raise NotImplementedError()


def test_model_gen():
    """test that the resulting model file is as expected and that
    the expected calls to fit the model, etc were made"""
    raise NotImplementedError()
