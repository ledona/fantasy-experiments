from collections import defaultdict
from typing import Literal, Iterable
from pprint import pprint
from datetime import datetime

import joblib
import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics
from tpot import TPOTRegressor
import pandas as pd

from fantasy_py.inference import SKLModel, TargetType
from fantasy_py import db


TrainTestData = tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]

_TargetType = tuple[Literal["stat", "calc"], str]


def load_data(
    filename: str, target: _TargetType, validation_season: int, seed=None
) -> TrainTestData:
    """Create train, test and validation data"""
    df = pd.read_csv(filename)

    one_hots = []
    if "extra:venue" in df:
        one_hots.append("extra:venue")
    if "pos" in df:
        one_hots.append("pos")
        df.pos = df.pos.astype(str)
    df = pd.get_dummies(df, columns=one_hots)

    df["extra:is_home"] = df["extra:is_home"].astype(int)

    feature_cols = [
        col
        for col in df
        if col == "pos" or col.startswith("extra") or ":recent" in col or ":std" in col
    ]

    train_test_df = df[df.season != validation_season]
    if len(train_test_df) == 0:
        raise ValueError("No training data!")

    target_col_name = ":".join(target)
    X = train_test_df[feature_cols]
    y = train_test_df[target_col_name]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=seed
    )

    validation_df = df[df.season == validation_season]
    if len(validation_df) == 0:
        raise ValueError("No validation data!")

    X_val = validation_df[feature_cols]
    y_val = validation_df[target_col_name]
    print(
        f"Training will use {len(feature_cols)} features, "
        f"{len(X_train)} training cases, "
        f"{len(X_test)} test cases, {len(X_val)} validation test cases",
    )

    return X_train, y_train, X_test, y_test, X_val, y_val


_AutomlType = Literal["tpot", "autosk"]


def train_test(
    type_: _AutomlType,
    model_name: str,
    target: _TargetType,
    tt_data: TrainTestData,
    seed: None | int,
    training_time: int,
    dt_trained: datetime,
):
    """
    train test and save a model

    training_time - max time to train in seconds

    returns the filepath to the model
    """
    print(f"Commencing {type_} fit with {training_time=}...")
    if type_ == "autosk":
        automl = autosklearn.regression.AutoSklearnRegressor(
            seed=seed, time_left_for_this_task=training_time, memory_limit=-1
        )
    elif type_ == "tpot":
        automl = TPOTRegressor(
            random_state=seed,
            max_time_mins=training_time / 60,
            verbosity=3,
        )
    else:
        raise NotImplementedError(f"automl type {type_} not recognized")

    (X_train, y_train, X_test, y_test, X_val, y_val) = tt_data
    automl.fit(X_train, y_train)

    if type_ == "autosk":
        print(automl.leaderboard())
        pprint(automl.show_models(), indent=4)
    else:
        pprint(automl.fitted_pipeline_)

    y_hat = automl.predict(X_test)
    r2 = round(sklearn.metrics.r2_score(y_test, y_hat), 3)
    mae = round(sklearn.metrics.mean_absolute_error(y_test, y_hat), 3)
    print(f"Test {r2=} {mae=}")

    y_hat_val = automl.predict(X_val)
    r2 = round(sklearn.metrics.r2_score(y_val, y_hat_val), 3)
    mae = round(sklearn.metrics.mean_absolute_error(y_val, y_hat_val), 3)
    print(f"Validation {r2=} {mae=}")

    filename = f"{model_name}-{type_}-{target[0]}:{target[1]}-{training_time}.{dt_trained.isoformat().rsplit('.', 1)[0]}.pkl"
    print(f"Exporting model to '{filename}'")
    if type_ == "autosk":
        joblib.dump(automl, filename)
    elif type_ == "tpot":
        joblib.dump(automl.fitted_pipeline_, filename)
    else:
        raise NotImplementedError(f"automl type {type_} not recognized")

    return filename


def create_fantasy_model(
    name: str,
    model_path: str,
    dt_trained: datetime,
    columns: Iterable[str],
    target: _TargetType,
    training_time,
    p_or_t: db.model.P_OR_T,
    recent_games: int,
    automl_type: _AutomlType,
    seed=None,
    recent_mean: bool = True,
    recent_explode: bool = True,
):
    """Create a model object based"""
    target_info = TargetType(target[0], p_or_t, target[1])
    features: dict[str, set] = defaultdict(set)
    for col in columns:
        type_, stat_name = col.split(":", 2)[:2]
        assert type_ in ["calc", "stat", "extra"]
        features[type_].add(stat_name)
    features_list_dict = {name: list(stats) for name, stats in features.items()}
    model = SKLModel(
        name,
        target_info,
        features_list_dict,
        dt_trained=dt_trained,
        data_def={
            "recent_games": recent_games,
            "recent_mean": recent_mean,
            "recent_explode": recent_explode,
        },
        parameters={
            "train_time": training_time,
            "seed": seed,
            "automl_type": automl_type,
        },
        trained_parameters={"regressor_path": model_path},
    )
    return model
