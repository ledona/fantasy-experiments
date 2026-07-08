import os
from typing import NamedTuple

import pandas as pd
from fantasy_py import DataNotAvailableException, DFSContestStyle, log
from fantasy_py.analysis.backtest.daily_fantasy import ModelFeatures, get_expected_bt_data_cols
from fantasy_py.betting import Contest
from sklearn.model_selection import train_test_split

_LOGGER = log.get_logger(__name__)


def load_csv(
    sport,
    service_: None | str,
    style: DFSContestStyle | str,
    contest_type: Contest | str,
    data_folder=".",
) -> pd.DataFrame:
    contest_type_name = contest_type if isinstance(contest_type, str) else contest_type.TYPE_NAME
    style_name = style if isinstance(style, str) else style.name
    services = [service_] if service_ is not None else ["fanduel", "draftkings", "yahoo"]

    dfs: list[pd.DataFrame] = []
    failed_filenames: list[str] = []
    for service in services:
        filename = f"{sport}-{service}-{style_name}-{contest_type_name}.csv"
        filepath = os.path.join(data_folder, filename)
        _LOGGER.info("loading '%s'", filepath)
        try:
            service_df = pd.read_csv(filepath)
            if len(services) > 1:
                service_df["service"] = service
            _LOGGER.info("for %s, %i rows of data loaded", filepath, len(service_df))
            dfs.append(service_df)
        except FileNotFoundError:
            failed_filenames.append(filepath)
    if len(dfs) == 0:
        raise DataNotAvailableException(f"Failed to find training data for {failed_filenames}")
    if len(failed_filenames) > 0:
        _LOGGER.info(
            "Failed to find following data files %s. Using what data was found for modeling.",
            failed_filenames,
        )

    df = pd.concat(dfs)
    nan_slate_rows = len(df.query("slate_id.isnull()"))
    nan_best_score_rows = len(df.query("top_possible_lineup_score.isnull()"))
    if nan_slate_rows > 0 or nan_best_score_rows > 0:
        orig_rows = len(df)
        df = df.dropna(subset=["slate_id", "top_possible_lineup_score"])
        _LOGGER.info(
            "Dropped %i rows. nan_slate_rows=%i nan_best_score_rows=%i. Remaining cases "
            "after drop = %i",
            orig_rows - len(df),
            nan_slate_rows,
            nan_best_score_rows,
            len(df),
        )
    if len(missing_last_winning_score_rows := df.query("last_winning_score == 0")) > 0:
        df = df.query("last_winning_score > 0")
        _LOGGER.info(
            "Dropped %i rows because last winning score is 0. %i rows remaining",
            len(missing_last_winning_score_rows),
            len(df),
        )

    if len(df) == 0:
        raise DataNotAvailableException("After filtering no data was left. see log for details")
    return df


class TrainTestData(NamedTuple):
    """storage for train/test data"""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_top: pd.Series
    y_test_top: pd.Series
    y_train_lws: pd.Series
    y_test_lws: pd.Series
    train_slate_ids: pd.Series
    test_slate_ids: pd.Series


def generate_train_test(
    input_df: pd.DataFrame,
    sport: str,
    style: DFSContestStyle,
    features: ModelFeatures,
    train_size: float = 0.75,
    random_state: None | int = None,
    drop_na_rows=False,
):
    """
    create regression train test data
    model_cols - if none then use all available columns
    drop_na_rows - default behavior is to only drop rows where top/lws score is not available, when true if any
        columns is NA a row will be dropped
    return (X-train, X-test, y-top-train, y-top-test, y-last-win-train, y-last-win-test)
    """
    expected_cols = get_expected_bt_data_cols(
        sport, style, features, False, True, set(input_df.columns)
    )
    df = input_df[expected_cols].assign(slate_id=input_df.slate_id)
    assert "service" not in df

    len_pre_na_drop = len(df)
    na_cols_to_test = None if drop_na_rows else ["top_winning_score", "last_winning_score"]
    df = df.dropna(subset=na_cols_to_test)
    if len(df) < len_pre_na_drop:
        _LOGGER.info(
            "Dropped %i of %i rows due to NaNs in %s",
            len_pre_na_drop - len(df),
            len_pre_na_drop,
            "any column" if na_cols_to_test is None else f"cols: {na_cols_to_test}",
        )
    if len(df) < 2:
        return None

    X = df.drop(columns=["top_winning_score", "last_winning_score", "slate_id"])
    y_top = df["top_winning_score"]
    y_lws = df["last_winning_score"]

    try:
        split_result = train_test_split(
            X, y_top, y_lws, df.slate_id, random_state=random_state, train_size=train_size
        )
    except ValueError as ex:
        _LOGGER.info("generate_train_test_split:: Error generating train test split", exc_info=ex)
        return None

    return TrainTestData(*split_result)
