import logging
import os

import pandas as pd
from fantasy_py.fantasy_types import ContestStyle
from fantasy_py.lineup.strategy import Contest
from sklearn.model_selection import train_test_split

_LOGGER = logging.getLogger(__name__)

_COLS_TO_IGNORE = {
    "date",
    "style",
    "type",
    "link",
    "entries",
    "slate_id",
    "top_score",
    "last_winning_score",
}


def load_csv(
    sport,
    service_: None | str,
    style: ContestStyle | str,
    contest_type: Contest | str,
    data_folder=".",
) -> pd.DataFrame:
    contest_type_name = contest_type if isinstance(contest_type, str) else contest_type.NAME
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
        raise FileNotFoundError(f"Failed to find {failed_filenames}")
    if len(failed_filenames) > 0:
        _LOGGER.info(
            "Failed to find following data files %s. Using what data was found for modeling.",
            failed_filenames,
        )

    df = pd.concat(dfs)
    nan_slate_rows = len(df.query("slate_id.isnull()"))
    nan_best_score_rows = len(df.query("`best-possible-score`.isnull()"))
    if nan_slate_rows > 0 or nan_best_score_rows > 0:
        orig_rows = len(df)
        df = df.dropna()
        _LOGGER.info(
            "Dropped %i rows. nan_slate_rows=%i nan_best_score_rows=%i. Remaining cases "
            "after drop = %i",
            orig_rows - len(df),
            nan_slate_rows,
            nan_best_score_rows,
            len(df),
        )

    return df


def generate_train_test(
    df: pd.DataFrame,
    train_size: float = 0.5,
    random_state: None | int = None,
    model_cols: None | set[str] = None,
    service_as_feature: bool = False,
) -> None | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    create regression train test data
    model_cols - if none then use all available columns
    return (X-train, X-test, y-top-train, y-top-test, y-last-win-train, y-last-win-test)
    """
    x_cols = []
    assert (
        (model_cols is None)
        or (isinstance(model_cols, set) and model_cols <= set(df.columns))
        or (isinstance(model_cols, str) and model_cols in set(df.columns))
    ), "Requested model columns not a subset of available data columns"

    for col in df.columns:
        if col in _COLS_TO_IGNORE:
            continue
        assert (
            "|" in col or col.startswith("team") or col in ["service", "best-possible-score"]
        ), f"Unexpected data column named '{col}'"

        if (model_cols is None) or col in model_cols:
            x_cols.append(col)

    len_pre_na_drop = len(df)
    if not service_as_feature and 'service' in df:
        df = df.drop(columns="service")
    df = df[df["top_score"].notna()]
    df = df[df["last_winning_score"].notna()]
    if len(df) < len_pre_na_drop:
        _LOGGER.info(
            "Dropped %i rows of %i due to NaNs in top_score or last_winning_score",
            len_pre_na_drop - len(df),
            len_pre_na_drop,
        )
    if len(df) < 2:
        return None

    X = df[x_cols]
    y_top = df["top_score"]
    y_last_win = df["last_winning_score"]

    try:
        sample_data = train_test_split(
            X, y_top, y_last_win, random_state=random_state, train_size=train_size
        )
    except ValueError as ex:
        _LOGGER.info("generate_train_test_split:: Error generating train test split", exc_info=ex)
        return None

    return sample_data
