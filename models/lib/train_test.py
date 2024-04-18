"""use this module's functions to train and evaluate models"""

import os
import platform
import re
from collections import defaultdict
from datetime import datetime
from glob import glob
from pprint import pprint
from typing import Literal, cast

import dateutil
import joblib
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

from fantasy_py import (
    SPORT_DB_MANAGER_DOMAIN,
    CLSRegistry,
    FantasyException,
    FeatureDict,
    PlayerOrTeam,
    UnexpectedValueError,
    dt_to_filename_str,
)
from fantasy_py.inference import Model, Performance, SKLModel, StatInfo, NNRegressor
from fantasy_py.sport import SportDBManager
from sklearn.dummy import DummyRegressor
from tpot import TPOTRegressor
from tpot.config import regressor_config_dict, regressor_config_dict_light

TrainTestData = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


class WildcardFilterFoundNothing(FantasyException):
    """raised if a wildcard feature filter did not match any columns"""


def _load_data(filename: str, include_position: bool | None, col_drop_filters: list[str] | None):
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

    cols_to_drop = []
    if col_drop_filters:
        regexps = []
        for filter_ in col_drop_filters:
            if "*" in filter_:
                regexps.append(re.compile(filter_.replace("*", ".*")))
                continue
            cols_to_drop.append(filter_)
        if len(regexps) > 0:
            for regexp in regexps:
                re_cols_to_drop = [col for col in df_raw if regexp.match(col)]
                if len(re_cols_to_drop) == 0:
                    raise WildcardFilterFoundNothing(
                        f"Filter '{regexp}' did not match any columns: {df_raw.columns}"
                    )
                cols_to_drop += re_cols_to_drop
        print(f"Dropping n={len(cols_to_drop)} columns: {sorted(cols_to_drop)}")
        df = df_raw.drop(columns=cols_to_drop)
    else:
        df = df_raw

    print(f"Include player position = {include_position}")
    one_hots = [
        col
        for col in df.columns
        if ":" in col and isinstance(df[col].iloc[0], str) and col not in cols_to_drop
    ]
    if "pos" in df:
        df.drop(columns="pos_id", inplace=True)
        if include_position:
            assert (
                col_drop_filters is None or "pos" not in col_drop_filters
            ), "conflicting request for pos and drop pos"
            one_hots.append("pos")
            df.pos = df.pos.astype(str)

    print(f"One-hot encoding features: {one_hots}")
    df = pd.get_dummies(df, columns=one_hots)

    one_hot_stats = (
        {"extra:venue": [col for col in df.columns if col.startswith("extra:venue_")]}
        if "extra:venue" in df
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
        for col in df.columns
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

    print(
        f"\nMISSING-DATA-REPORT cases={len(df)} warning_threshold={warning_threshold * 100:.02f}%"
    )
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
    seed: int | None,
    include_position: None | bool = None,
    col_drop_filters: None | list[str] = None,
    filtering_query: None | str = None,
    missing_data_threshold=0,
):
    """
    Create train, test and validation data

    target: tuple[stat type, stat name]
    include_position: If not None a 'pos' column is required in the loaded\
        data and will be included/excluded based on this argument. If None\
        and 'pos' is in the loaded data, an exception is raised
    cols_drop_filters: list of features to remove from data. '*'\
        will be wildcard matched, columns are dropped after filtering_query\
        is applied and one-hot encoding is performed
    filtering_query: query to execute (using dataframe.query) to filter\
        for rows in the input data. Executed before one-hot and column drops
    missing_data_threshold: warn about feature columns where data is not\
        found for more than this percentage of cases.E.g. 0 = warn in any data is missing\
        .25 = warn if > 25% of data is missing
    """
    target_col_name = ":".join(target)
    print(f"Target column name set to '{target_col_name}'")

    df_raw, df, one_hot_stats = _load_data(filename, include_position, col_drop_filters)
    if filtering_query:
        df = df.query(filtering_query)
        print(f"Filter '{filtering_query}' dropped {len(df_raw) - len(df)} rows")
    feature_cols = [
        col
        for col in df.columns
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
    if len(features_not_found := set(feature_cols) - set(train_test_df.columns)) > 0:
        raise ValueError(
            f"Following requested feature models not found in data: {features_not_found}"
        )
    X = train_test_df[feature_cols]
    if target_col_name not in train_test_df:
        available_targets = [col for col in train_test_df.columns if len(col.split(":")) == 2]
        raise ValueError(
            f"Target feature '{target_col_name}' not found in data. "
            f"Available targets are {available_targets}"
        )
    y = train_test_df[target_col_name]

    _missing_feature_data_report(X, missing_data_threshold)
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
        f"{len(X_test)} test cases, {len(X_val)} validation test cases from {validation_season=}",
    )

    return (
        df_raw,
        cast(TrainTestData, (X_train, y_train, X_test, y_test, X_val, y_val)),
        one_hot_stats,
    )


def _infer_imputes(train_df: pd.DataFrame, team_target: bool):
    """
    returns - a dict mapping column names to impute value to use, None if no
        imputation needed (i.e. no missing values)
    """
    impute_values = {
        Model.impute_key_for_feature_name(col, team_target): round(train_df[col].median(), 2)
        for col in train_df.columns
        if (":std-mean" in col or col.startswith("extra:"))
    }
    if len(impute_values) == 0:
        print(
            "No season to date features found in data. "
            "Impute data will not be included in model."
        )
        return None
    return impute_values


ArchitectureType = Literal["tpot", "tpot-light", "dummy", "auto-xgb", "nn"]


def train_test(
    type_: ArchitectureType,
    model_name: str,
    target: StatInfo,
    tt_data: TrainTestData,
    dest_dir: str,
    **model_init_kwargs,
) -> tuple[str, Performance, datetime]:
    """
    train, test and save a model to a pickle

    training_time: max time to train in seconds
    returns the filepath to the model
    """
    dt_trained = datetime.now()
    print(f"Fitting {model_name=} using {type_}")
    if type_ == "tpot":
        model = TPOTRegressor(
            verbosity=3,
            **model_init_kwargs,
        )
    elif type_ == "tpot-light":
        model = TPOTRegressor(
            verbosity=3,
            config_dict=regressor_config_dict_light,
            **model_init_kwargs,
        )
    elif type_ == "auto-xgb":
        model = TPOTRegressor(
            config_dict={"xgboost.XGBRegressor": regressor_config_dict["xgboost.XGBRegressor"]},
        )
    elif type_ == "dummy":
        model = DummyRegressor(**model_init_kwargs)
    elif type_ == "nn":
        model = NNRegressor(**model_init_kwargs)
    else:
        raise NotImplementedError(f"architecture {type_} not recognized")

    (X_train, y_train, X_test, y_test, X_val, y_val) = tt_data
    model.fit(X_train, y_train)

    if type_.startswith("tpot"):
        pprint(model.fitted_pipeline_)
    elif type_ == "dummy":
        print("Dummy fitted")
    elif type_ == "auto-xgb":
        print("XGB fitted")
    elif type_ == "nn":
        print("NN fitted")
    else:
        raise NotImplementedError(f"model type {type_} not recognized")

    y_hat = model.predict(X_test)
    r2_test = round(float(sklearn.metrics.r2_score(y_test, y_hat)), 3)
    mae_test = round(float(sklearn.metrics.mean_absolute_error(y_test, y_hat)), 3)
    print(f"Test {r2_test=} {mae_test=}")

    y_hat_val = model.predict(X_val)
    r2_val = round(float(sklearn.metrics.r2_score(y_val, y_hat_val)), 3)
    mae_val = round(float(sklearn.metrics.mean_absolute_error(y_val, y_hat_val)), 3)
    print(f"Validation {r2_val=} {mae_val=}")

    filepath = os.path.join(
        dest_dir,
        f"{model_name}-{type_}-{target[0]}.{target[1]}.{dt_to_filename_str(dt_trained)}.pkl",
    )
    print(f"Exporting model artifact to '{filepath}'")
    if type_ in ("dummy", "auto-xgb"):
        joblib.dump(model, filepath)
    elif isinstance(model, TPOTRegressor):
        joblib.dump(model.fitted_pipeline_, filepath)
    elif isinstance(model, NNRegressor):
        raise NotImplementedError()
    else:
        raise NotImplementedError(f"model type {type_} not recognized")

    return filepath, {"r2": r2_val, "mae": mae_val}, dt_trained


def _create_fantasy_model(
    name: str,
    model_artifact_path: str,
    algo_type: ArchitectureType,
    dt_trained: datetime,
    train_df: pd.DataFrame,
    target: StatInfo,
    performance: Performance,
    p_or_t: PlayerOrTeam,
    recent_games: int,
    training_seasons: list[int],
    target_pos: None | list[str],
    training_pos: None | list[str],
    model_params: dict[str, str | int],
    one_hot_stats: dict[str, list[str]] | None = None,
    recent_mean: bool = True,
    recent_explode: bool = True,
    only_starters: bool | None = None,
) -> Model:
    """Create a model object based"""
    print(f"Creating fantasy model for {name=}")
    assert one_hot_stats is None or list(one_hot_stats.keys()) == ["extra:venue"]
    target_info = StatInfo(target[0], p_or_t, target[1])
    include_pos = False
    features: FeatureDict = defaultdict(set)
    columns = train_df.columns
    sport_abbr = name.split("-", 1)[0]
    db_manager = cast(
        SportDBManager, CLSRegistry.get_class(SPORT_DB_MANAGER_DOMAIN, sport_abbr.lower())
    )
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
            if extra_name not in db_manager.EXTRA_STATS:
                possible_1_hot_extras = [
                    name for name in db_manager.EXTRA_STATS if extra_name.startswith(name)
                ]
                if len(possible_1_hot_extras) == 0:
                    raise ValueError(
                        f"Unrecognized extra stat '{extra_name}'. For sport={sport_abbr}, "
                        f"valid extra stats are {db_manager.EXTRA_STATS}"
                    )
                if len(possible_1_hot_extras) > 1:
                    raise ValueError(
                        f"Extra stat '{extra_name}' could be a one hot of multiple "
                        f"{sport_abbr} extra stats. "
                        "Can't figure out which of the following extra stats to use: "
                        f"{possible_1_hot_extras}"
                    )
                print(
                    f"One hotted extra stat '{extra_name}' assigned to original "
                    f"extra stat '{possible_1_hot_extras[0]}'"
                )
                extra_name = possible_1_hot_extras[0]

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
    imputes = _infer_imputes(train_df, p_or_t == PlayerOrTeam.TEAM)
    uname = platform.uname()
    model = SKLModel(
        name,
        target_info,
        features,
        dt_trained=dt_trained,
        trained_on_uname=uname._asdict(),
        training_data_def=data_def,
        parameters={**model_params, "algo_type": algo_type},
        trained_parameters={"regressor_path": model_artifact_path},
        performance=performance,
        player_positions=target_pos,
        input_cols=columns.to_list(),
        impute_values=imputes,
    )

    return model


def model_and_test(
    name: str,
    validation_season: int,
    tt_data,
    target,
    algo_type: ArchitectureType,
    p_or_t,
    recent_games,
    training_seasons,
    ml_kwargs,
    target_pos: None | list[str],
    training_pos,
    dest_dir,
    reuse_most_recent: bool,
):
    """create or load a model and test it"""
    model = None
    if reuse_most_recent:
        model_filename_pattern = ".".join([name, target[1], algo_type, "*", "model"])
        most_recent_model: tuple[datetime, str] | None = None
        for filename in glob(os.path.join(dest_dir, model_filename_pattern)):
            model_dt = dateutil.parser.parse(filename.split(".")[3])
            if (most_recent_model is None) or (most_recent_model[0] < model_dt):
                most_recent_model = (model_dt, filename)

        if most_recent_model is not None:
            final_model_filepath = most_recent_model[1]
            print(f"Reusing model at '{final_model_filepath}'")
            model = Model.load(final_model_filepath)

    if model is None:
        final_model_filepath = os.path.join(
            dest_dir, ".".join([name, target[1], algo_type, dt_to_filename_str(), "model"])
        )

        model_artifact_path, performance, dt_trained = train_test(
            algo_type,
            name,
            target,
            tt_data,
            dest_dir,
            **ml_kwargs,
        )
        performance["season"] = validation_season

        model = _create_fantasy_model(
            name,
            model_artifact_path,
            algo_type,
            dt_trained,
            tt_data[0],
            target,
            performance,
            p_or_t,
            recent_games,
            training_seasons,
            target_pos,
            training_pos,
            ml_kwargs,
        )

        model.dump(final_model_filepath, overwrite=not reuse_most_recent)
        print(f"Model file saved to '{final_model_filepath}'")

    return model
