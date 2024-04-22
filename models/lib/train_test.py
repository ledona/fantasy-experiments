"""use this module's functions to train and evaluate models"""

import math
import os
import platform
import re
from collections import defaultdict
from datetime import datetime
from glob import glob
from pprint import pprint
from tempfile import gettempdir
from typing import Literal, Type, cast

import dateutil
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn.metrics
import sklearn.model_selection
import torch
from fantasy_py import (
    SPORT_DB_MANAGER_DOMAIN,
    CLSRegistry,
    DataNotAvailableException,
    FantasyException,
    FeatureDict,
    InvalidArgumentsException,
    PlayerOrTeam,
    UnexpectedValueError,
    dt_to_filename_str,
    log,
)
from fantasy_py.inference import Model, NNModel, NNRegressor, Performance, SKLModel, StatInfo
from fantasy_py.sport import SportDBManager
from sklearn.dummy import DummyRegressor
from tpot import TPOTRegressor
from tpot.config import regressor_config_dict, regressor_config_dict_light

TrainTestData = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]

_LOGGER = log.get_logger(__name__)


class WildcardFilterFoundNothing(FantasyException):
    """raised if a wildcard feature filter did not match any columns"""


def _load_data(
    filename: str,
    include_position: bool | None,
    col_drop_filters: list[str] | None,
    limit: int | None,
    validation_season: int,
):
    """
    validation_season: make sure that there is some validation data, even if\
        there is a limit. If there is a limit and no validation data is retrieved\
        when pulling data up to the limit then perform a subsequent retrieval of\
        at most 10% of the limit from the validation season
    """
    if filename.endswith(".csv"):
        df_raw = pd.read_csv(filename, nrows=limit)
        if len(df_raw.query("season == @validation_season")) == 0:
            raise DataNotAvailableException(
                "Limited load of data from csv file failed to retrieve data from "
                "validation season. Try again without limit, or try a different "
                "the validation season."
            )
    elif filename.endswith(".pq") or filename.endswith(".parquet"):
        if limit is not None:
            pf = pq.ParquetFile(filename)
            first_n_rows = next(pf.iter_batches(batch_size=limit))
            df_raw = pa.Table.from_batches([first_n_rows]).to_pandas()
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
        _LOGGER.info("Dropping n=%i columns: %s", len(cols_to_drop), sorted(cols_to_drop))
        df = df_raw.drop(columns=cols_to_drop)
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
            assert (
                col_drop_filters is None or "pos" not in col_drop_filters
            ), "conflicting request for pos and drop pos"
            one_hots.append("pos")
            df.pos = df.pos.astype(str)

    _LOGGER.info("One-hot encoding features: %s", one_hots)
    df = pd.get_dummies(df, columns=one_hots)

    # one_hot_stats = (
    #     {"extra:venue": [col for col in df.columns if col.startswith("extra:venue_")]}
    #     if "extra:venue" in df
    #     else None
    # )

    if "extra:is_home" in df:
        df["extra:is_home"] = df["extra:is_home"].astype(int)

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
    limit: int | None = None,
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

    returns tuple of (raw data, {train, test and validation data}, stats that are one-hot-transformed)
    """
    target_col_name = ":".join(target)
    _LOGGER.info("Target column name set to '%s'", target_col_name)

    df_raw, df, one_hot_stats = _load_data(
        filename, include_position, col_drop_filters, limit, validation_season
    )
    if filtering_query:
        df = df.query(filtering_query)
        _LOGGER.info("Filter '%s' dropped %i rows", filtering_query, len(df_raw) - len(df))
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

    _LOGGER.info("Final feature cols: %s", sorted(feature_cols))

    train_test_df = df[df.season != validation_season]
    if len(train_test_df) == 0:
        _LOGGER.warning("No training data found for non-validation seasons. Using validation data.")
        train_test_df = df
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
        raise ValueError(f"No validation data from season {validation_season} retrieved!")

    X_val = validation_df[feature_cols]
    y_val = validation_df[target_col_name]
    _LOGGER.info(
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
    df = train_df.fillna(0)
    impute_values = {
        Model.impute_key_for_feature_name(col, team_target): round(df[col].median(), 2)
        for col in train_df.columns
        if (":std-mean" in col or col.startswith("extra:"))
    }
    if len(impute_values) == 0:
        _LOGGER.info(
            "No season to date features found in data. "
            "Impute data will not be included in model."
        )
        return None
    return impute_values


ArchitectureType = Literal["tpot", "tpot-light", "dummy", "auto-xgb", "nn"]


def _create_model_obj(
    arch: ArchitectureType, model_init_kwargs: dict, x: pd.DataFrame, y: pd.Series, model_filebase
):
    fit_addl_args = None
    if arch == "tpot":
        model = TPOTRegressor(
            verbosity=3,
            **model_init_kwargs,
        )
    elif arch == "tpot-light":
        model = TPOTRegressor(
            verbosity=3,
            config_dict=regressor_config_dict_light,
            **model_init_kwargs,
        )
    elif arch == "auto-xgb":
        model = TPOTRegressor(
            config_dict={"xgboost.XGBRegressor": regressor_config_dict["xgboost.XGBRegressor"]},
        )
    elif arch == "dummy":
        model = DummyRegressor(**model_init_kwargs)
    elif arch == "nn":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hidden_size = 2 ** int(math.log2(len(x.columns)))
        input_size = len(x.columns)
        if "checkpoint_dir" not in model_init_kwargs:
            checkpoint_dir = os.path.join(
                gettempdir(), "fantasy-nn-checkpoints", model_filebase + "-" + dt_to_filename_str()
            )
            _LOGGER.info("Creating nn checkpoint directory '%s'", checkpoint_dir)
            os.mkdir(checkpoint_dir)
        else:
            checkpoint_dir = model_init_kwargs.pop("checkpoint_dir")
        model = NNRegressor(
            input_size, hidden_size=hidden_size, checkpoint_dir=checkpoint_dir, **model_init_kwargs
        ).to(device)
        fit_addl_args = (x, y)
    else:
        raise NotImplementedError(f"architecture {arch} not recognized")

    return model, fit_addl_args


def train_test(
    type_: ArchitectureType,
    model_name: str,
    target: StatInfo,
    tt_data: TrainTestData,
    dest_dir: str,
    model_filebase: str,
    **model_init_kwargs,
) -> tuple[str, Performance, datetime]:
    """
    train, test and save a model to a pickle

    model_filebase: basename for model and artifact files
    training_time: max time to train in seconds
    returns the filepath to the model
    """
    dt_trained = datetime.now()
    _LOGGER.info("Fitting model_name=%s using type=%s", model_name, type_)

    (X_train, y_train, X_test, y_test, X_val, y_val) = tt_data
    model, fit_addl_args = _create_model_obj(
        type_, model_init_kwargs, X_test, y_test, model_filebase
    )
    model = model.fit(X_train, y_train, *(fit_addl_args or []))

    if type_.startswith("tpot"):
        _LOGGER.info("TPOT fitted")
        pprint(cast(TPOTRegressor, model).fitted_pipeline_)
    elif type_ in ("dummy", "auto-xgb", "nn"):
        _LOGGER.info("%s fitted", type_)
    else:
        raise NotImplementedError(f"model type {type_} not recognized")

    y_hat = model.predict(X_test)
    r2_test = round(float(sklearn.metrics.r2_score(y_test, y_hat)), 3)
    mae_test = round(float(sklearn.metrics.mean_absolute_error(y_test, y_hat)), 3)
    _LOGGER.info("Test r2_test=%f mae_test=%f", r2_test, mae_test)

    y_hat_val = model.predict(X_val)
    r2_val = round(float(sklearn.metrics.r2_score(y_val, y_hat_val)), 3)
    mae_val = round(float(sklearn.metrics.mean_absolute_error(y_val, y_hat_val)), 3)

    _LOGGER.info("Validation r2_val=%f mae_val=%f", r2_val, mae_val)

    artifact_filebase = (
        model_filebase
        or f"{model_name}-{type_}-{target[0]}.{target[1]}.{dt_to_filename_str(dt_trained)}"
    )
    artifact_filebase_path = os.path.join(dest_dir, artifact_filebase)
    _LOGGER.info("Exporting model artifact to '%s'", artifact_filebase_path)
    if type_ in ("dummy", "auto-xgb"):
        artifact_filebase_path += ".pkl"
        joblib.dump(model, artifact_filebase_path)
    elif isinstance(model, TPOTRegressor):
        artifact_filebase_path += ".pkl"
        joblib.dump(model.fitted_pipeline_, artifact_filebase_path)
    elif isinstance(model, NNRegressor):
        artifact_filebase_path += ".pt"
        torch.save(model, artifact_filebase_path)
    else:
        raise NotImplementedError(f"model type {type_} not recognized")

    return artifact_filebase_path, {"r2": r2_val, "mae": mae_val}, dt_trained


def _get_model_cls(arch: ArchitectureType) -> Type[Model]:
    if arch == "nn":
        return NNModel
    return SKLModel


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
    """Create a model object"""
    _LOGGER.info("Creating fantasy model for '%s'", name)
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
                _LOGGER.info(
                    "One hotted extra stat '%s' assigned to original extra stat '%s'",
                    extra_name,
                    possible_1_hot_extras[0],
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
    model_cls = _get_model_cls(algo_type)
    model = model_cls(
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
    model_dest_filename: str | None = None,
):
    """
    create or load a model and test it
    model_dest_filename: name of the file to write the model to. default is to use\
        the default model filename pattern based on the model name
    reuse_most_recent: do not create a new model if one already exists that follows\
        the default model filenaming pattern. If an existing model exists, use the\
        most recently created version
    """
    model = None
    if reuse_most_recent:
        if model_dest_filename is not None:
            raise InvalidArgumentsException(
                "reuse_most_recent cannot be used with a model_dest_filename"
            )
        model_filename_pattern = ".".join([name, target[1], algo_type, "*", "model"])
        most_recent_model: tuple[datetime, str] | None = None
        for filebase_name in glob(os.path.join(dest_dir, model_filename_pattern)):
            model_dt = dateutil.parser.parse(filebase_name.split(".")[3])
            if (most_recent_model is None) or (most_recent_model[0] < model_dt):
                most_recent_model = (model_dt, filebase_name)

        if most_recent_model is not None:
            final_model_filepath = most_recent_model[1]
            _LOGGER.info("Reusing model at '%s'", final_model_filepath)
            model = Model.load(final_model_filepath)

    if model is None:
        filebase_name = model_dest_filename or ".".join(
            [name, target[1], algo_type, dt_to_filename_str()]
        )
        if filebase_name.endswith(".model"):
            filebase_name = filebase_name.rsplit(".", 1)[0]

        model_artifact_path, performance, dt_trained = train_test(
            algo_type,
            name,
            target,
            tt_data,
            dest_dir,
            filebase_name,
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

        final_model_filepath = os.path.join(dest_dir, filebase_name + ".model")
        model.dump(final_model_filepath, overwrite=not reuse_most_recent)
        _LOGGER.info("Model file saved to '%s'", final_model_filepath)

    return model
