"""use this module's functions to train and evaluate models"""

from collections import defaultdict
import os
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

from fantasy_py import FantasyException, UnexpectedValueError, PlayerOrTeam, FeatureDict
from fantasy_py.inference import SKLModel, StatInfo, Model, Performance


_TPOT_JOBS = 2

TrainTestData = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


class WildcardFilterFoundNothing(FantasyException):
    """raised if a wildcard feature filter did not match any columns"""


def _load_data(filename: str, include_position: bool | None):
    print(f"Loading data from '{filename}'")
    if filename.endswith(".csv"):
        df_raw = pd.read_csv(filename)
    elif filename.endswith(".pq") or filename.endswith(".parquet"):
        df_raw = pd.read_parquet(filename)
    else:
        raise NotImplementedError(
            f"Don't know how to load data files with extension {filename.rsplit('.', 1)[-1]}"
        )

    if include_position is not None and "pos" not in df_raw:
        raise UnexpectedValueError(
            "Column 'pos' not found in data, 'include_position' must be None!"
        )
    if include_position is None and "pos" in df_raw:
        raise UnexpectedValueError(
            "Column 'pos' found in data, 'include_position' kwarg is required!"
        )

    print(f"Include player position set to {include_position}")
    one_hots = [col for col in df_raw if ":" in col and isinstance(df_raw[col].iloc[0], str)]
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


def _missing_feature_data_report(df: pd.DataFrame, warning_threshold):
    counts = df.count()
    counts.name = "valid-data"
    counts.index.name = "feature-name"
    missing_data_df = pd.DataFrame(counts).reset_index()
    missing_data_df["%-NA"] = missing_data_df["valid-data"].map(lambda x: 100 * (1 - x / len(df)))
    missing_data_df["%-valid"] = missing_data_df["valid-data"].map(lambda x: 100 * x / len(df))
    warning_df = missing_data_df.query("`%-NA` > (@warning_threshold * 100)")

    print(f"\nMISSING-DATA-REPORT case={len(df)} warning_threshold={warning_threshold * 100:.02f}%")
    if len(warning_df) == 0:
        print(f"All features have less than {warning_threshold * 100:.02f}% missing values")
        return

    print(
        f"{len(counts)} of {len(df.columns)} features have >{warning_threshold * 100:.02f}% missing values."
    )

    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
    ):
        print(
            warning_df.to_string(
                index=False, formatters={"%-NA": "{:.02f}%".format, "%-valid": "{:.02f}%".format}
            )
        )


def load_data(
    filename: str,
    target: tuple[str, str],
    validation_season: int,
    include_position: None | bool = None,
    seed=None,
    col_drop_filters: None | list[str] = None,
    filtering_query: None | str = None,
    missing_data_threshold=0,
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
    missing_data_threshold - warn about feature columns where data is not
        found for more than this percentage of cases.E.g. 0 = warn in any data is missing
        .25 = warn if > 25% of data is missing
    """
    target_col_name = ":".join(target)
    print(f"Target column name set to '{target_col_name}'")

    df_raw, df, one_hot_stats = _load_data(filename, include_position)
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
        assert len(cols_to_drop) > 0, f"No columns to drop from {col_drop_filters=} {regexps=}"
        print(f"Dropping the following {len(cols_to_drop)} columns: ", cols_to_drop)
        X = X.drop(columns=cols_to_drop)
    else:
        cols_to_drop = None

    _missing_feature_data_report(X, missing_data_threshold)
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


def _infer_imputes(train_df: pd.DataFrame, team_target: bool):
    """
    returns - a dict mapping column names to impute value to use, None if no
        imputation needed (i.e. no missing values)
    """
    impute_values = {
        Model.impute_key_for_feature_name(col, team_target): round(train_df[col].median(), 2)
        for col in train_df
        if ":std-mean" in col
    }
    if len(impute_values) == 0:
        print(
            "No season to date features found in data. "
            "Impute data will not be included in model."
        )
        return None
    return impute_values


_AutomlType = Literal["tpot", "autosk", "dummy"]


def train_test(
    type_: _AutomlType,
    model_name: str,
    target: StatInfo,
    tt_data: TrainTestData,
    seed: None | int,
    training_time: int,
) -> tuple[str, Performance, datetime]:
    """
    train test and save a model ti a pickle

    training_time - max time to train in seconds

    returns the filepath to the model
    """
    dt_trained = datetime.now()
    print(f"Commencing training for {model_name=} {type_} fit with {training_time=}...")
    if type_ == "autosk":
        raise NotImplementedError(
            "disabled until autosk uses more up to date version of sklearn 2023.08.06"
        )
        automl = autosklearn.regression.AutoSklearnRegressor(
            seed=seed, time_left_for_this_task=training_time, memory_limit=-1
        )
    elif type_ == "tpot":
        automl = TPOTRegressor(
            random_state=seed, max_time_mins=training_time / 60, verbosity=3, n_jobs=_TPOT_JOBS
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
    elif type_ == "tpot":
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

    filename = f"{model_name}-{type_}-{target[0]}.{target[1]}.{dt_trained.isoformat().replace(':', '').rsplit('.', 1)[0]}.pkl"
    print(f"Exporting model artifact to '{filename}'")
    if type_ in ("autosk", "dummy"):
        joblib.dump(automl, filename)
    elif type_ == "tpot":
        joblib.dump(automl.fitted_pipeline_, filename)
    else:
        raise NotImplementedError(f"automl type {type_} not recognized")

    return filename, {"r2": r2_val, "mae": mae_val}, dt_trained


def create_fantasy_model(
    name: str,
    model_artifact_path: str,
    dt_trained: datetime,
    train_df: pd.DataFrame,
    target: StatInfo,
    training_time,
    automl_type: _AutomlType,
    performance: Performance,
    p_or_t: PlayerOrTeam,
    recent_games: int,
    training_seasons: list[int],
    one_hot_stats: dict[str, list[str]] | None = None,
    seed=None,
    recent_mean: bool = True,
    recent_explode: bool = True,
    only_starters: bool | None = None,
    target_pos: None | list[str] = None,
    training_pos: None | list[str] = None,
) -> Model:
    """Create a model object based"""
    print(f"Creating fantasy model for {name=}")
    assert one_hot_stats is None or list(one_hot_stats.keys()) == ["extra:venue"]
    target_info = StatInfo(target[0], p_or_t, target[1])
    include_pos = False
    features: FeatureDict = defaultdict(set)
    columns = train_df.columns
    for col in columns:
        if col.startswith("pos_"):
            include_pos = True
            continue
        col_split = col.split(":")

        assert len(col_split) >= 2 and col_split[0] in ["calc", "stat", "extra"]

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

        assert col_split[0] in ["calc", "stat"]
        if len(col_split) == 4:
            if col_split[-1] == "player-team":
                features["player_team_" + col_split[0]].add(col_split[1])
                continue
            if col_split[-1] == "opp-team":
                features["opp_team_" + col_split[0]].add(col_split[1])
                continue
        if len(col_split) == 3:
            features[col_split[0]].add(col_split[1])
            continue

        raise UnexpectedValueError(f"Unknown feature type for data column named col='{col}'")

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
        features,
        dt_trained=dt_trained,
        data_def=data_def,
        parameters={
            "train_time": training_time,
            "seed": seed,
            "automl_type": automl_type,
        },
        trained_parameters={"regressor_path": model_artifact_path},
        performance=performance,
        player_positions=target_pos,
        input_cols=columns.to_list(),
        impute_values=_infer_imputes(train_df, p_or_t == PlayerOrTeam.TEAM),
    )

    return model


def model_and_test(
    name,
    validation_season,
    tt_data,
    target,
    training_time,
    automl_type,
    *args,
    reuse_existing=False,
    raw_df=None,
    **kwargs,
):
    """create or load a model and test it"""
    model_filename = ".".join([name, target[1], automl_type, "model"])
    print(f"Model filename = '{model_filename}'")
    if not reuse_existing or not os.path.exists(model_filename):
        if reuse_existing:
            print("Reuse failed, existing model not found")
        model_artifact_path, performance, dt_trained = train_test(
            automl_type, name, target, tt_data, kwargs["seed"], training_time
        )
        performance["season"] = validation_season

        model = create_fantasy_model(
            name,
            model_artifact_path,
            dt_trained,
            tt_data[0],
            target,
            training_time,
            automl_type,
            performance,
            *args,
            **kwargs,
        )
        model_filepath = model.dump(
            ".".join([name, target[1], automl_type, "model"]),
            validate_reg_filepath=False,
        )
        print(f"Model file saved to '{model_filepath}'")
    else:
        print("Reusing existing model...")
        model = Model.load(model_filename)

    if raw_df is not None:
        try:
            model.predict(raw_df.sample(10))
            print("model post testing model successful...")
        except Exception as ex:
            print(f"post prediction testing failed! '{type(ex).__name__}':")
            print(traceback.format_exc())
    else:
        print("not post testing model ...")

    return model
