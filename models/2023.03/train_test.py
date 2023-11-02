from collections import defaultdict
from typing import Literal
import traceback
from pprint import pprint
from datetime import datetime
import re

import joblib
# import autosklearn.regression
import sklearn.model_selection
import sklearn.metrics
from sklearn.dummy import DummyRegressor
from tpot import TPOTRegressor
import pandas as pd

from fantasy_py.inference import SKLModel, StatInfo, Model, Performance
from fantasy_py import FantasyException, UnexpectedValueError, PlayerOrTeam


TrainTestData = tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]


class WildcardFilterFoundNothing(FantasyException):
    """raised if a wildcard feature filter did not match any columns"""


def load_csv(filename: str, include_position: bool | None):
    print(f"Loading data from '{filename}'")
    df_raw = pd.read_csv(filename)

    if include_position is not None and "pos" not in df_raw:
        raise UnexpectedValueError(
            "Column 'pos' not found in data, 'include_position' must be None!"
        )
    elif include_position is None and "pos" in df_raw:
        raise UnexpectedValueError(
            "Column 'pos' found in data, 'include_position' kwarg is required!"
        )

    print(f"Include player position set to {include_position}")
    one_hots = []
    if "extra:venue" in df_raw:
        one_hots.append("extra:venue")
    if "pos" in df_raw:
        df_raw.drop(columns="pos_id", inplace=True)
        if include_position:
            one_hots.append("pos")
            df_raw.pos = df_raw.pos.astype(str)

    print(f"One-hot encoding features: {one_hots}")
    df = pd.get_dummies(df_raw, columns=one_hots)

    one_hot_stats = (
        {"extra:venue": [col for col in df if col.startswith("extra:venue_")]}
        if "extra:venue" in df_raw
        else None
    )

    if "extra:is_home" in df:
        df["extra:is_home"] = df["extra:is_home"].astype(int)

    return df_raw, df, one_hot_stats


def infer_feature_cols(df: pd.DataFrame, include_position: bool):
    """
    figure out what the feature columns for training/inference will be based
    on the columns in df
    """
    return [
        col
        for col in df
        if (col.startswith("pos_") and include_position is True and col != "pos_id")
        or col.startswith("extra")
        or ":recent" in col
        or ":std" in col
    ]


def load_data(
    filename: str,
    target: tuple[str, str],
    validation_season: int,
    include_position: None | bool = None,
    seed=None,
    col_drop_filters: None | list[str] = None,
    filtering_query: None | str = None,
):
    """
    Create train, test and validation data

    target - tuple[stat type, stat name]
    include_position - If not None a 'pos' column is required in the loaded
        data and will be included/excluded based on this argument. If None
        and 'pos' is in the loaded data, an exception is raised
    cols_drop_filters - list of features to remove from data. '*'
        will be wildcard matched, columns are dropped after filtering_query
        is applied and one-hot encoding is performed
    filtering_query - query to execute (using dataframe.query) to filter
        for rows in the input data. Executed before one-hot and column drops
    """
    target_col_name = ":".join(target)
    print(f"Target column name set to '{target_col_name}'")

    df_raw, df, one_hot_stats = load_csv(filename, include_position)
    if filtering_query:
        df = df.query(filtering_query)
        print(f"Filter '{filtering_query}' dropped {len(df_raw) - len(df)} rows")
    feature_cols = [
        col
        for col in df
        if col != target_col_name
        and (
            (col.startswith("pos_") and include_position is True)
            or col.startswith("extra")
            or ":recent" in col
            or ":std" in col
        )
    ]

    train_test_df = df[df.season != validation_season]
    if len(train_test_df) == 0:
        raise ValueError("No training data!")
    X = train_test_df[feature_cols]
    y = train_test_df[target_col_name]

    if col_drop_filters:
        cols_to_drop = []
        regexps = []
        for filter_ in col_drop_filters:
            if "*" in filter_:
                regexps.append(re.compile(filter_.replace("*", ".*")))
                continue
            cols_to_drop.append(filter_)
        if len(regexps) > 0:
            for regexp in regexps:
                re_cols_to_drop = [col for col in feature_cols if regexp.match(col)]
                if len(re_cols_to_drop) == 0:
                    raise WildcardFilterFoundNothing(
                        f"Filter '{regexp}' did not match any columns: {feature_cols}"
                    )
                cols_to_drop += re_cols_to_drop
        assert (
            len(cols_to_drop) > 0
        ), f"No columns to drop from {col_drop_filters=} {regexps=}"
        print(f"Dropping the following {len(cols_to_drop)} columns: ", cols_to_drop)
        X = X.drop(columns=cols_to_drop)
    else:
        cols_to_drop = None

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=seed
    )

    validation_df = df[df.season == validation_season]
    if len(validation_df) == 0:
        raise ValueError("No validation data!")

    X_val = validation_df[feature_cols]
    if cols_to_drop is not None:
        X_val = X_val.drop(columns=cols_to_drop)
    y_val = validation_df[target_col_name]
    print(
        f"Training will use {len(feature_cols)} features, "
        f"{len(X_train)} training cases, "
        f"{len(X_test)} test cases, {len(X_val)} validation test cases from {validation_season=}",
    )

    return df_raw, (X_train, y_train, X_test, y_test, X_val, y_val), one_hot_stats


def infer_imputes(train_df: pd.DataFrame):
    """
    returns - a dict mapping column names to impute value to use, None if no
        imputation needed (i.e. no missing values)
    """
    impute_values = {
        col.rsplit(":", 1)[0]: round(train_df[col].median(), 2)
        for col in train_df
        if col.endswith("std-mean")
    }
    return impute_values if len(impute_values) > 0 else None


_AutomlType = Literal["tpot", "autosk", "dummy"]


def train_test(
    type_: _AutomlType,
    model_name: str,
    target: StatInfo,
    tt_data: TrainTestData,
    seed: None | int,
    training_time: int,
    dt_trained: datetime,
) -> tuple[str, Performance]:
    """
    train test and save a model

    training_time - max time to train in seconds

    returns the filepath to the model
    """
    print(f"Commencing training for {model_name=} {type_} fit with {training_time=}...")
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
    elif type_ == "dummy":
        automl = DummyRegressor()
    else:
        raise NotImplementedError(f"automl type {type_} not recognized")

    (X_train, y_train, X_test, y_test, X_val, y_val) = tt_data
    automl.fit(X_train, y_train)

    if type_ == "autosk":
        print(automl.leaderboard())
        pprint(automl.show_models(), indent=4)
    elif type == "tpot":
        pprint(automl.fitted_pipeline_)
    else:
        print("Dummy fitted")

    y_hat = automl.predict(X_test)
    r2_test = round(sklearn.metrics.r2_score(y_test, y_hat), 3)
    mae_test = round(sklearn.metrics.mean_absolute_error(y_test, y_hat), 3)
    print(f"Test {r2_test=} {mae_test=}")

    y_hat_val = automl.predict(X_val)
    r2_val = round(sklearn.metrics.r2_score(y_val, y_hat_val), 3)
    mae_val = round(sklearn.metrics.mean_absolute_error(y_val, y_hat_val), 3)
    print(f"Validation {r2_val=} {mae_val=}")

    filename = f"{model_name}-{type_}-{target[0]}:{target[1]}.{dt_trained.isoformat().rsplit('.', 1)[0]}.pkl"
    print(f"Exporting model to '{filename}'")
    if type_ in ("autosk", "dummy"):
        joblib.dump(automl, filename)
    elif type_ == "tpot":
        joblib.dump(automl.fitted_pipeline_, filename)
    else:
        raise NotImplementedError(f"automl type {type_} not recognized")

    return filename, {"r2": r2_val, "mae": mae_val}


def create_fantasy_model(
    name: str,
    model_path: str,
    dt_trained: datetime,
    train_df: pd.DataFrame,
    target: StatInfo,
    training_time,
    p_or_t: PlayerOrTeam,
    recent_games: int,
    automl_type: _AutomlType,
    performance: Performance,
    training_seasons: list[int],
    one_hot_stats: dict[str, list[str]] | None = None,
    seed=None,
    recent_mean: bool = True,
    recent_explode: bool = True,
    only_starters: bool | None = None,
    target_pos: None | list[str] = None,
    training_pos: None | list[str] = None,
    raw_df: pd.DataFrame | None = None,
) -> Model:
    """Create a model object based"""
    print(f"Creating fantasy model for {name=}")
    assert one_hot_stats is None or list(one_hot_stats.keys()) == ["extra:venue"]
    target_info = StatInfo(target[0], p_or_t, target[1])
    include_pos = False
    features: dict[str, set] = defaultdict(set)
    columns = train_df.columns
    for col in columns:
        if col.startswith("pos_"):
            include_pos = True
            continue
        col_split = col.split(":")
        assert len(col_split) >= 2 and col_split[0] in ["calc", "stat", "extra"]
        if col_split[0] in ["calc", "stat"]:
            features[col_split[0]].add(col_split[1])
            continue
        if col_split[0] == "extra":
            if col_split[1].startswith("venue_"):
                assert len(col_split) == 2
                extra_type = "current_extra"
                extra_name = "venue"
            else:
                extra_type = "hist_extra" if len(col_split) > 2 else "current_extra"
                extra_name = col_split[1]
            features[extra_type].add(extra_name)
            continue

        raise UnexpectedValueError(
            f"Unknown feature type for data column named '{col}'"
        )

    features_list_dict = {name: list(stats) for name, stats in features.items()}
    data_def: dict = {
        "recent_games": recent_games,
        "recent_mean": recent_mean,
        "recent_explode": recent_explode,
        "include_pos": include_pos,
        "seasons": training_seasons,
    }
    if only_starters is not None:
        data_def["only_starters"] = only_starters
    if training_pos is not None:
        data_def["training_pos"] = training_pos
    model = SKLModel(
        name,
        target_info,
        features_list_dict,
        dt_trained=dt_trained,
        training_data_def=data_def,
        parameters={
            "train_time": training_time,
            "seed": seed,
            "automl_type": automl_type,
        },
        trained_parameters={"regressor_path": model_path},
        performance=performance,
        player_positions=target_pos,
        input_cols=columns.to_list(),
        impute_values=infer_imputes(train_df),
    )

    if raw_df is not None:
        try:
            model.predict(raw_df.sample(10))
            print("post testing model predict successful...")
        except Exception as ex:
            print(f"post prediction testing failed! '{type(ex).__name__}':")
            print(traceback.format_exc())
    else:
        print("not post testing model ...")
    return model
