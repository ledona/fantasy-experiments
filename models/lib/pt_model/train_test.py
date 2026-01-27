"""use this module's functions to train and evaluate models"""

import os
import platform
import re
from collections import defaultdict
from datetime import datetime
from glob import glob
from pprint import pformat
from typing import Literal, cast

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn.metrics
import sklearn.model_selection
from dateutil import parser as du_parser
from fantasy_py import (
    SPORT_DB_MANAGER_DOMAIN,
    CLSRegistry,
    DataNotAvailableException,
    ExtraFeatureType,
    FantasyException,
    FeatureType,
    PlayerOrTeam,
    UnexpectedValueError,
    dt_to_filename_str,
    log,
    now,
)
from fantasy_py.inference import (
    AutogluonModel,
    NNModel,
    NNRegressor,
    PerformanceDict,
    PTPredictModel,
    SKLModel,
    StatInfo,
)
from fantasy_py.sport import SportDBManager
from ledona import slack
from sklearn.dummy import DummyRegressor

from .autogluon import AutoGluonWrapper
from .nn import NNWrapper
from .tpot import TPOTWrapper

TrainTestData = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]


_LOGGER = log.get_logger(__name__)


class WildcardFilterFoundNothing(FantasyException):
    """raised if a wildcard feature filter did not match any columns"""


def _load_data_local(
    filename: str,
    include_position: bool | None,
    col_drop_filters: list[str] | None,
    limit: int | None,
    validation_season: int,
    target_col_name: str,
    original_model_cols: set[str] | None,
):
    """
    validation_season: make sure that there is some validation data, even if\
        there is a limit. If there is a limit and no validation data is retrieved\
        when pulling data up to the limit then perform a subsequent retrieval of\
        at most 10% of the limit from the validation season
    col_drop_filters: list of regex filters to identify columns to drop. only\
        applied to feature columns and never matches target col. \
        these are columns that start with a feature type prefix
    original_model_cols: not none means that we are retraining a model and so\
        have an expected list of final input cols
    """
    if filename.endswith(".csv"):
        df_raw = pd.read_csv(filename, nrows=limit)
        if len(df_raw.query("season == @validation_season")) == 0:
            raise DataNotAvailableException(
                "No data returned from limited load of data from csv file failed "
                "to retrieve data from validation season. Try again without limit, "
                "or try a different the validation season."
            )
    elif filename.endswith(".parquet") or filename.endswith(".pq"):
        if limit is not None:
            pf = pq.ParquetFile(filename)
            first_n_rows = next(pf.iter_batches(batch_size=limit))
            df_raw = cast(pd.DataFrame, pa.Table.from_batches([first_n_rows]).to_pandas())
            if len(df_raw.query("season == @validation_season")) == 0:
                df_validation = pd.read_parquet(
                    filename, filters=[("season", "=", validation_season)]
                )
                df_raw = pd.concat([df_raw, df_validation])
                _LOGGER.info(
                    "Loaded %i rows from '%s' but no validation data from "
                    "season %i was loaded. Additional load for "
                    "validation data was done resulting in total (limit+validation) of "
                    "%i rows.",
                    limit,
                    filename,
                    validation_season,
                    len(df_raw),
                )
        else:
            df_raw = pd.read_parquet(filename)
    else:
        raise NotImplementedError(
            f"Don't know how to load data files with extension {filename.rsplit('.', 1)[-1]}"
        )

    if target_col_name not in df_raw:
        # available targets are columns that have a single colon
        available_targets = [col for col in df_raw.columns if len(col.split(":")) == 2]
        raise UnexpectedValueError(
            f"Target feature '{target_col_name}' not found in data. "
            f"Available targets are {available_targets}"
        )

    if include_position is True and "pos" not in df_raw:
        raise UnexpectedValueError(
            "Column 'pos' not found in data, 'include_position' must be None!"
        )
    if include_position is None and "pos" in df_raw:
        raise UnexpectedValueError(
            "Column 'pos' found in data, 'include_position' kwarg is required!"
        )

    cols_to_drop: set[str] = set()
    if col_drop_filters:
        feature_regex = r"^(stat|extra|calc):.*"
        for filter_ in col_drop_filters:
            re_cols_to_drop: set[str] = {
                col
                for col in df_raw.columns
                if re.match(filter_, col)
                and re.match(feature_regex, col)
                and col != target_col_name
            }
            if len(re_cols_to_drop) == 0:
                _LOGGER.error(
                    "Filter '%s' did not match any input columns: %s",
                    filter_,
                    sorted(df_raw.columns),
                )
            else:
                _LOGGER.info(
                    "Column drop filter '%s' matched to %i feature cols",
                    filter_,
                    len(re_cols_to_drop),
                )
            cols_to_drop |= re_cols_to_drop
        _LOGGER.info(
            "Dropping n=%i of %i columns",  # : %s",
            len(cols_to_drop),
            len([col for col in df_raw.columns if ":" in col]),
            # sorted(cols_to_drop),
        )
        df = df_raw.drop(columns=list(cols_to_drop))
    else:
        df = df_raw

    _LOGGER.info("Include player position = %s", include_position)

    # one-hot encode anything where the first value is a string
    one_hots = [
        col
        for col in df.columns
        if ":" in col and isinstance(df[col].iloc[0], str) and col not in cols_to_drop
    ]
    if "pos" in df:
        df.drop(columns="pos_id", inplace=True)
        if include_position:
            assert col_drop_filters is None or "pos" not in col_drop_filters, (
                "conflicting request for pos and drop pos"
            )
            one_hots.append("pos")
            df.pos = df.pos.astype(str)

    _LOGGER.info("One-hot encoding features: %s", one_hots)
    df = pd.get_dummies(df, columns=one_hots)

    if "extra:is_home" in df:
        df["extra:is_home"] = df["extra:is_home"].astype(int)

    if original_model_cols is not None:
        final_cols = [
            col
            for col in df.columns
            if ":" not in col or col in original_model_cols or col == target_col_name
        ]
        df = df[final_cols]

    if df[target_col_name].hasnans:
        pre_drop_len = len(df)
        df = df.query(f"`{target_col_name}`.notna()")
        _LOGGER.info(
            "Dropped %i rows where target col '%s' was nan.",
            pre_drop_len - len(df),
            target_col_name,
        )

    return df_raw, df, one_hots


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


def _missing_feature_data_report(df: pd.DataFrame, warning_threshold, fail_threshold, skip_report):
    assert warning_threshold < fail_threshold
    counts = df.count()
    counts.name = "valid-data"
    counts.index.name = "feature-name"
    missing_data_df = pd.DataFrame(counts).reset_index()
    missing_data_df = missing_data_df.assign(
        **{
            "%-NA": missing_data_df["valid-data"].map(lambda x: 100 * (1 - x / len(df))),
            "%-valid": missing_data_df["valid-data"].map(lambda x: 100 * x / len(df)),
        }
    )
    warning_df = missing_data_df.query("`%-NA` > (@warning_threshold * 100)")

    if not skip_report:
        print("\n   -----  MISSING-DATA-REPORT  -----")
        print(f"cases={len(df)} warning_threshold={warning_threshold * 100:.02f}%")
        if len(warning_df) == 0:
            print(f"All features have less than {warning_threshold * 100:.02f}% missing values")

    if len(warning_df) == 0:
        return

    if not skip_report:
        print(
            f"{len(warning_df)} of {len(df.columns)} features have "
            f">{warning_threshold * 100:.02f}% missing values."
        )

        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.max_colwidth", None
        ):
            print(
                warning_df.to_string(
                    index=False,
                    formatters={"%-NA": "{:.02f}%".format, "%-valid": "{:.02f}%".format},
                )
            )
    fail_df = missing_data_df.query("`%-NA` > (@fail_threshold * 100)")
    if len(fail_df) > 0:
        raise DataNotAvailableException(
            "The following features were below the fail threshold of "
            f"{fail_threshold}: {fail_df['feature-name'].to_list()}"
        )


def _summarize_final_feature_cols(
    df: pd.DataFrame,
    target_col_name: str,
    one_hot_stats: list[str],
    skip_data_reports: bool,
    include_position: bool,
):
    feature_cols = sorted(
        col
        for col in df.columns
        if col != target_col_name
        and (
            (col.startswith("pos_") and include_position)
            or col.startswith("extra")
            or ":recent" in col
            or ":std" in col
        )
    )

    _LOGGER.info("Final feature cols n=%i", len(feature_cols))

    if skip_data_reports:
        return feature_cols

    features: dict = {}
    for col in feature_cols:
        col_strs = col.split(":")
        if len(one_hot_stats) > 0:
            is_one_hot = False
            one_hot = None
            for one_hot in one_hot_stats:
                if col.startswith(one_hot):
                    is_one_hot = True
                    break
            if is_one_hot:
                assert one_hot is not None
                if not (
                    (len(col_strs) == 1 and col.startswith("pos_"))
                    or len(col_strs) == 2
                    or (len(col_strs) == 3 and col_strs[2].startswith("opp-team_"))
                ):
                    raise UnexpectedValueError(
                        f"Could not parse one-hot-feature '{col}'. "
                        "One hot stats should be clean or opp-team."
                    )
                row_dict = {"one-hot": True}
                if len(col_strs) == 3:
                    assert col_strs[2].startswith("opp-team_"), ""
                    name = ":".join(col_strs[:2])
                    row_dict["opp-team"] = True
                else:
                    row_dict["clean"] = True
                    name = one_hot
                if name in features:
                    features[name].update(row_dict)
                else:
                    features[name] = row_dict
                continue
        feature_str = ":".join(col_strs[:2])
        if feature_str not in features:
            features[feature_str] = {}
        if len(col_strs) == 2:
            features[feature_str]["clean"] = True
            continue
        features[feature_str].update({attr: True for attr in col_strs[2:]})

    df = pd.DataFrame.from_dict(features, orient="index").fillna(False)
    max_recent_explode = 0
    for col in df.columns:
        if not col.startswith("recent-"):
            continue
        post_fix = col.split("-")[1]
        if not post_fix.isdecimal():
            continue
        max_recent_explode = max(int(post_fix), max_recent_explode)
    df = (
        df[sorted(df.columns)]
        .drop(columns=[f"recent-{n}" for n in range(1, max_recent_explode)])
        .sort_index()
    )

    print()
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        1000,
    ):
        print(df)
    return feature_cols


def load_data(
    filename: str,
    target: tuple[str, str] | str,
    validation_season: int,
    seed: int | None,
    include_position: None | bool = None,
    col_drop_filters: None | list[str] = None,
    filtering_query: None | str = None,
    missing_data_warn_threshold=0.0,
    missing_data_fail_threshold=0.5,
    limit: int | None = None,
    expected_cols: None | set[str] = None,
    skip_data_reports=False,
):
    """
    Create train, test and validation data

    target: tuple[stat type, stat name] or 'stat-type:stat-name'
    include_position: If not None a 'pos' column is required in the loaded\
        data and will be included/excluded based on this argument. If None\
        and 'pos' is in the loaded data, an exception is raised
    cols_drop_filters: list of features to remove from data. '*'\
        will be wildcard matched, columns are dropped after filtering_query\
        is applied and one-hot encoding is performed
    filtering_query: query to execute (using dataframe.query) to filter\
        for rows in the input data. Executed before one-hot and column drops.\
        Data is ALWAYS filtered for notna of target
    missing_data_threshold: warn about feature columns where data is not\
        found for more than this percentage of cases.E.g. 0 = warn in any data is missing\
        .25 = warn if > 25% of data is missing

    returns tuple of (raw data, {train, test and validation data}, one-hot-transformed stats)
    """
    target_col_name = target if isinstance(target, str) else ":".join(target)
    _LOGGER.info("Target column name set to '%s'", target_col_name)

    df_raw, df, one_hot_stats = _load_data_local(
        filename,
        include_position,
        col_drop_filters,
        limit,
        validation_season,
        target_col_name,
        expected_cols,
    )
    if filtering_query:
        try:
            df = df.query(filtering_query)
        except pd.errors.UndefinedVariableError:
            _LOGGER.error(
                "The requested filter likely includes columns that are not in the dataframe"
            )
            raise
        _LOGGER.info("Filter '%s' dropped %i rows", filtering_query, len(df_raw) - len(df))

    feature_cols = _summarize_final_feature_cols(
        df, target_col_name, one_hot_stats, skip_data_reports, include_position is True
    )

    train_test_df = df[df.season != validation_season]
    if len(train_test_df) == 0:
        _LOGGER.warning("No training data found for non-validation seasons. Using validation data.")
        train_test_df = df
    if len(features_not_found := set(feature_cols) - set(train_test_df.columns)) > 0:
        raise UnexpectedValueError(
            f"Following requested feature models not found in data: {features_not_found}"
        )

    X = train_test_df[feature_cols]
    y = train_test_df[target_col_name]

    _missing_feature_data_report(
        X, missing_data_warn_threshold, missing_data_fail_threshold, skip_data_reports
    )
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=seed
    )

    validation_df = df[df.season == validation_season]
    if len(validation_df) == 0:
        raise UnexpectedValueError(f"No validation data from season {validation_season} retrieved!")

    X_val = validation_df[feature_cols]
    y_val = validation_df[target_col_name]
    _LOGGER.info(
        "Data contains %i features, %i training cases, "
        "%i test cases, %i validation test cases from validation_season=%i",
        len(feature_cols),
        len(X_train),
        len(X_test),
        len(X_val),
        validation_season,
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
    df = train_df.fillna(0)
    impute_values = {
        PTPredictModel.impute_key_for_feature_name(col, team_target): round(df[col].median(), 2)
        for col in sorted(train_df.columns)
        if (":std-mean" in col or col.startswith("extra:"))
    }
    if len(impute_values) == 0:
        _LOGGER.info(
            "No season to date features found in data. Impute data will not be included in model."
        )
        return None
    return impute_values


AlgorithmType = Literal["autogluon", "tpot", "tpot-light", "dummy", "nn", "xgboost"]
"""machine learning algorithm used for model selection and training"""


def _instantiate_regressor(
    algorithm: AlgorithmType, model_params: dict, x: pd.DataFrame, y: pd.Series, model_filebase
):
    """returns (model-obj, fit-args, fit-kwargs)"""
    if algorithm == "autogluon":
        model = AutoGluonWrapper(
            model_filebase,
            verbosity=model_params["verbose"],
            preset=model_params["ag:preset"],
            time_limit=model_params["max_time_mins"] * 60,
        )
        return model

    if algorithm.startswith("tpot"):
        model = TPOTWrapper(tpot_light=algorithm == "tpot-light", **model_params)
        return model

    if algorithm == "dummy":
        model = DummyRegressor(strategy=model_params.get("dmy:strategy"))
        return model

    if algorithm == "nn":
        model = NNWrapper(x, y, model_filebase, **model_params)
        return model

    raise NotImplementedError(f"{algorithm=} not recognized")


def _train_test(
    algo: AlgorithmType,
    model_name: str,
    target: tuple[FeatureType, str],
    tt_data: TrainTestData,
    dest_dir: str,
    model_filebase: str,
    **model_params,
) -> tuple[str, PerformanceDict, datetime, dict | None]:
    """
    train, test and save a model to a pickle

    model_filebase: basename for model and artifact files
    training_time: max time to train in seconds
    returns tuple[mode-filepath, model-performance, train-dt, dict-describing-training+model]
    """
    dt_trained = now()
    _LOGGER.info("Fitting model_name=%s using type=%s", model_name, algo)

    (X_train, y_train, X_test, y_test, X_val, y_val) = tt_data
    if len(X_train.columns) == 0 or len(X_train) == 0:
        raise DataNotAvailableException(
            "Cannot train because width or length of training data is 0. "
            f"width={len(X_train.columns)} len={len(X_train)}"
        )
    model = _instantiate_regressor(algo, model_params, X_test, y_test, model_filebase)
    model.fit(X_train, y_train)
    assert model is not None
    training_desc_info: dict = {
        "time_to_fit": str(now() - dt_trained),
        "n_train_cases": len(X_train),
        "n_test_cases": len(X_test),
        "n_validation_cases": len(X_val),
    }

    if algo.startswith("tpot") or algo == "autogluon":
        assert isinstance(model, (TPOTWrapper, AutoGluonWrapper))
        model.update_training_desc_info(training_desc_info)
        model.log_fitted_model()
    elif algo in ("dummy", "nn"):
        _LOGGER.success("%s fitted", algo)
        if algo == "nn":
            training_desc_info["epochs_trained"] = cast(NNRegressor, model).epochs_trained
    else:
        raise NotImplementedError(f"model type {algo} not recognized")

    y_hat_train = model.predict(X_train)
    r2_train = float(sklearn.metrics.r2_score(y_train, y_hat_train))
    mae_train = float(sklearn.metrics.mean_absolute_error(y_train, y_hat_train))
    _LOGGER.info("Train      r2=%g mae=%g", round(r2_train, 6), round(mae_train, 6))

    y_hat_test = model.predict(X_test)
    r2_test = float(sklearn.metrics.r2_score(y_test, y_hat_test))
    mae_test = float(sklearn.metrics.mean_absolute_error(y_test, y_hat_test))
    _LOGGER.info("Test       r2=%g mae=%g", round(r2_test, 6), round(mae_test, 6))

    y_hat_val = model.predict(X_val)
    r2_val = float(sklearn.metrics.r2_score(y_val, y_hat_val))
    mae_val = float(sklearn.metrics.mean_absolute_error(y_val, y_hat_val))
    _LOGGER.info("Validation r2=%g mae=%g", round(r2_val, 6), round(mae_val, 6))

    artifact_filebase = (
        model_filebase or f"{model_name}.{algo}.{target[1]}.{dt_to_filename_str(dt_trained)}"
    )
    artifact_filebase_path = os.path.join(dest_dir, artifact_filebase)

    if algo == "dummy":
        artifact_filepath = artifact_filebase_path + ".pkl"
        joblib.dump(model, artifact_filepath)
    elif algo in ("nn", "autogluon") or algo.startswith("tpot"):
        assert isinstance(model, (NNWrapper, TPOTWrapper, AutoGluonWrapper))
        artifact_filepath = model.save_artifact(artifact_filebase_path)
    else:
        raise NotImplementedError(f"model {algo=} not recognized")

    _LOGGER.info("Exported model artifact to '%s'", artifact_filepath)
    performance: PerformanceDict = {
        "r2_val": r2_val,
        "mae_val": mae_val,
        "r2_test": r2_test,
        "mae_test": mae_test,
        "r2_train": r2_train,
        "mae_train": mae_train,
    }
    return artifact_filepath, performance, dt_trained, training_desc_info


def _get_model_cls(algo: AlgorithmType):
    if algo.startswith("tpot") or algo == "dummy":
        return SKLModel
    if algo == "autogluon":
        return AutogluonModel
    if algo == "nn":
        return NNModel
    raise NotImplementedError(f"Don't know what model class to use for {algo=}")


_EXTRA_HIST_PART_PATTERN = re.compile("std-mean|recent-(mean|[1-9])")
"""test pattern for the historic part of an extra stat's name"""


def _infer_extra_stat_name_type(
    name_parts: list[str],
) -> tuple[str, ExtraFeatureType]:
    """
    figure out the extra stat name and type, and validate the type
    returns (name, type) where type = current_extra | hist_extra
    """
    if name_parts[1].startswith("venue_"):
        assert len(name_parts) == 2
        return "venue", "current_extra"

    extra_name = name_parts[1]
    if len(name_parts) == 2:
        if _EXTRA_HIST_PART_PATTERN.match(name_parts[1]) or name_parts[1] in (
            "player-team",
            "opp-team",
        ):
            raise UnexpectedValueError(
                f"{name_parts=} for parts of len=2, "
                "the second part should not match hist or team part patterns"
            )
        return extra_name, "current_extra"

    if len(name_parts) == 3:
        if _EXTRA_HIST_PART_PATTERN.match(name_parts[2]):
            extra_type = "hist_extra"
        elif name_parts[2].startswith("player-team"):
            extra_type = "current_extra"
        elif name_parts[2].startswith("opp-team"):
            extra_type = "current_opp_team_extra"
        else:
            raise UnexpectedValueError(
                f"{name_parts=} do not recognize the 3rd part of the name. "
                "3rd part should define hist or type"
            )
        return extra_name, extra_type

    if len(name_parts) != 4:
        raise UnexpectedValueError(f"{name_parts=} number of name parts should be 2, 3 or 4")
    if not _EXTRA_HIST_PART_PATTERN.match(name_parts[2]):
        raise UnexpectedValueError(f"{name_parts=} 3rd part should define hist")

    if name_parts[3] == "player-team":
        extra_type = "hist_extra"
    elif name_parts[3] == "opp-team":
        extra_type = "hist_opp_team_extra"
    else:
        raise UnexpectedValueError(f"{name_parts=} last part should define team")

    return extra_name, extra_type


def _create_fantasy_model(
    name: str,
    sport: str,
    model_artifact_path: str,
    dt_trained: datetime,
    training_features_df: pd.DataFrame,
    target: tuple[FeatureType, str],
    performance: PerformanceDict,
    p_or_t: PlayerOrTeam,
    recent_games: int,
    training_seasons: list[int],
    target_pos: None | list[str],
    training_pos: None | list[str],
    limit: int | None,
    model_params: dict[str, str | int],
    model_info: None | dict,
    one_hot_stats: dict[str, list[str]] | None = None,
    recent_mean: bool = True,
    recent_explode: bool = True,
    only_starters: bool | None = None,
) -> PTPredictModel:
    """Create a model object"""
    _LOGGER.info("Creating fantasy model for '%s'", name)
    assert one_hot_stats is None or list(one_hot_stats.keys()) == ["extra:venue"]
    target_info = StatInfo(target[0], p_or_t, target[1])
    include_pos = False
    features_sets: dict[FeatureType, set[str]] = defaultdict(set)
    columns = training_features_df.columns.to_list()
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
            extra_name, extra_type = _infer_extra_stat_name_type(col_split)
            if extra_name not in db_manager.EXTRA_STATS:
                possible_1_hot_extras = [
                    name for name in db_manager.EXTRA_STATS if extra_name.startswith(name)
                ]
                if len(possible_1_hot_extras) == 0:
                    raise UnexpectedValueError(
                        f"Unrecognized extra stat '{extra_name}'. For sport={sport_abbr}, "
                        f"valid extra stats are {db_manager.EXTRA_STATS}"
                    )
                if len(possible_1_hot_extras) > 1:
                    raise UnexpectedValueError(
                        f"Extra stat '{extra_name}' could be a one hot of multiple "
                        f"{sport_abbr} extra stats. "
                        "Can't figure out which of the following extra stats to use: "
                        f"{possible_1_hot_extras}"
                    )
                _LOGGER.info(
                    "One hotted extra stat '%s' assigned to original extra stat '%s'",
                    extra_name,
                    possible_1_hot_extras[0],
                )
                extra_name = possible_1_hot_extras[0]

            features_sets[extra_type].add(extra_name)
            continue

        assert col_split[0] in ["calc", "stat"]
        if len(col_split) == 4:
            if col_split[-1] == "player-team":
                features_sets[cast(FeatureType, "player_team_" + col_split[0])].add(col_split[1])
                continue
            if col_split[-1] == "opp-team":
                features_sets[cast(FeatureType, "opp_team_" + col_split[0])].add(col_split[1])
                continue
        if len(col_split) == 3:
            features_sets[cast(FeatureType, col_split[0])].add(col_split[1])
            continue

        raise UnexpectedValueError(f"Unknown feature type for data column named col='{col}'")

    data_def: dict = {
        "recent_games": recent_games,
        "recent_mean": recent_mean,
        "recent_explode": recent_explode,
        "include_pos": include_pos,
        "seasons": training_seasons,
    }
    if limit is not None:
        data_def["limit"] = limit
    if only_starters is not None:
        data_def["only_starters"] = only_starters
    if training_pos is not None:
        data_def["training_pos"] = training_pos
    imputes = _infer_imputes(training_features_df, p_or_t.is_team)
    uname = platform.uname()
    model_cls = _get_model_cls(cast(AlgorithmType, model_params["algorithm"]))
    model = model_cls(
        name,
        sport,
        target_info,
        features_sets,
        dt_trained=dt_trained,
        trained_on_uname=uname._asdict(),
        training_data_def=data_def,
        parameters=model_params,
        trained_parameters={"regressor_path": model_artifact_path},
        performance=performance,
        player_positions=target_pos,
        input_cols=columns,
        impute_values=imputes,
        desc_info=model_info,
    )

    return model


ModelFileFoundMode = Literal["reuse", "overwrite", "create-w-ts"]
"""
'reuse' = If a model already exists at the derived or defined destination \
    filepath, model training will be skipped.

'overwrite' = Create a new model, if one exists at the expected target filepath\
    then overwrite it.

'create-w-ts' = Create a new model, if a model file already exists at the expected\
    destination filepath then write the new model to a new filepath that\
    includes a timestamp
"""


def _reuse_model_helper(
    model_dest_filename: str | None, dest_dir: str, name: str, target, algorithm: AlgorithmType
):
    """can an existing model be reused? if so return it."""
    if model_dest_filename is not None:
        dest_filename_w_ext = (
            (model_dest_filename + ".model") if not model_dest_filename.endswith(".model") else ""
        )
        final_model_filepath = os.path.join(dest_dir, dest_filename_w_ext)
        if os.path.isfile(final_model_filepath):
            _LOGGER.success("Reusing model at '%s'", final_model_filepath)
            return PTPredictModel.load(final_model_filepath)

    model_filename_pattern = ".".join([name, target[1], algorithm, "*", "model"])
    most_recent_model: tuple[datetime, str] | None = None
    for filebase_name in glob(os.path.join(dest_dir, model_filename_pattern)):
        model_dt = du_parser.parse(filebase_name.split(".")[-2])
        if (most_recent_model is None) or (most_recent_model[0] < model_dt):
            most_recent_model = (model_dt, filebase_name)

    if most_recent_model is not None:
        final_model_filepath = most_recent_model[1]
        _LOGGER.success("Reusing model at '%s'", final_model_filepath)
        return PTPredictModel.load(final_model_filepath)

    return None


def model_and_test(
    name: str,
    sport: str,
    validation_season: int,
    tt_data,
    target: tuple[FeatureType, str],
    algorithm: AlgorithmType,
    p_or_t,
    recent_games,
    training_seasons,
    model_params,
    target_pos: None | list[str],
    training_pos,
    dest_dir: str,
    mode: ModelFileFoundMode,
    limit: int | None,
    model_dest_filename: str | None = None,
    data_src_params: dict | None = None,
):
    """
    train or load a model, test it, then wrap it and export it for use

    model_dest_filename: name of the file to write the model to. default is to use\
        the default model filename pattern based on the model name
    mode: See ModelFileFoundMode
    data_src_params: parameters describing the data source for model training
    """
    model = (
        _reuse_model_helper(model_dest_filename, dest_dir, name, target, algorithm)
        if mode == "reuse"
        else None
    )

    if model is None:
        filebase_name = model_dest_filename or ".".join(
            [name, target[1], algorithm, dt_to_filename_str()]
        )
        if filebase_name.endswith(".model"):
            filebase_name = filebase_name.rsplit(".", 1)[0]

        final_model_filepath = os.path.join(dest_dir, filebase_name) + ".model"
        if os.path.isfile(final_model_filepath):
            if mode == "overwrite":
                _LOGGER.info(
                    "A new model will be written to '%s'. This will overwriting an existing model",
                    final_model_filepath,
                )
            else:
                existing_model_filepath = final_model_filepath
                filebase_name += "." + dt_to_filename_str()
                final_model_filepath = os.path.join(dest_dir, filebase_name) + ".model"
                _LOGGER.info(
                    "A model already exists at '%s'. A new model will be saved to '%s'",
                    existing_model_filepath,
                    final_model_filepath,
                )
        else:
            _LOGGER.info("A new model will be saved to '%s'", final_model_filepath)

        model_artifact_path, performance, dt_trained, addl_info = _train_test(
            algorithm,
            name,
            target,
            tt_data,
            dest_dir,
            filebase_name,
            **model_params,
        )
        performance["season_val"] = validation_season
        addl_params = {
            **model_params,
            **(data_src_params or {}),
            "algorithm": algorithm,
        }

        model = _create_fantasy_model(
            name,
            sport,
            model_artifact_path,
            dt_trained,
            tt_data[0],
            target,
            performance,
            p_or_t,
            recent_games,
            training_seasons,
            target_pos,
            training_pos,
            limit,
            addl_params,
            addl_info,
        )

        model.dump(final_model_filepath, overwrite=mode == "overwrite")
        slack.send_slack(f"Training done for {name}\n\nPerformance\n{pformat(model.performance)}")

    return model
