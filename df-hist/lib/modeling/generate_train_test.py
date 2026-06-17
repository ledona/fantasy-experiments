import os
from typing import Literal, cast

import pandas as pd
from fantasy_py import DataNotAvailableException, DFSContestStyle, UnexpectedValueError, log
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
        df = df.dropna()
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


_DESCRIPTIVE_COLS = {
    "date",
    "style",
    "type",
    "link",
    "slate_id",
}


ModelFeatures = Literal["all", "legacy", "202606"]
"""
feature set to use for models
legacy = pre-202606 model features
202606 = upgraded feature set
all = all available features (i.e. legacy + 202606)
"""


def test_for_expected_cols(
    inf_df: pd.DataFrame,
    sport,
    style: DFSContestStyle,
    unexpected_mode: Literal["drop", "fail"] = "fail",
    features: ModelFeatures = "all",
    include_descriptive_cols=True,
):
    """
    test for and reorder the columns
    unexpected_mode: fail=raise an exception, drop=drop unexpected cols and return a new df
    include_descriptive_cols: if true include things like date, slate_id, etc. False returns
        a dataframe just with features and target variables
    """
    df = inf_df

    # start with the targets
    expected_cols = [
        "top_winning_score",
        "last_winning_score",
        "top_possible_lineup_score",
        "top_rational_lineup_score",
        "top_possible_minus_rational",
        "low_cost_high_value_player_count",
    ]

    if include_descriptive_cols:
        expected_cols = [*_DESCRIPTIVE_COLS, expected_cols]

    if style == DFSContestStyle.CLASSIC:
        # everything gets this
        expected_cols.append("team_count")
        if features in ("all", "legacy"):
            # old features
            expected_cols += [
                "total_score",
                "team_med",
                "team-70.0th_pctl",
                "top3-total",
                "top_players_scoring_diff_n",
                "top_players_scoring_diff_pctl",
            ]
    elif style == DFSContestStyle.SHOWDOWN:
        if features in ("all", "202606"):
            # new features
            expected_cols += [
                "contest_entries",
                "game-total_score",  # same as total score, added because total score is not needed in new classic
                "game-margin_of_victor",
            ]
    else:
        raise NotImplementedError(f"Unhandled {style=}")

    if features in ("all", "202606"):
        # new sport specific features
        if sport == "mlb":
            expected_cols += ["high-team-score"]
            if style == DFSContestStyle.SHOWDOWN:
                expected_cols += ["winning-team-WH-allowed", "losing-team-WH-allowed"]
        elif sport == "nfl":
            if style == DFSContestStyle.SHOWDOWN:
                expected_cols.append("td-to-yardage")
            expected_cols.append("top-possible-lineup-DEF+K-score%")
        elif sport == "nhl":
            if style == DFSContestStyle.CLASSIC:
                expected_cols.append("3-player-line-goals%")
            elif style == DFSContestStyle.SHOWDOWN:
                # goalie stats
                expected_cols += ["winning-team-saves", "losing-team-saves"]
        elif sport == "nba":
            expected_cols += ["low_cost_high_use"]
        else:
            raise NotImplementedError(f"unhandled {sport=}")

    positional_cols = [
        col for col in df.columns if col.endswith("|70.0th-pctl-dfs") or col.endswith("|med-dfs")
    ]

    if features in ("all", "legacy"):
        expected_cols += positional_cols

    fail_msgs = []

    missing_cols = set(expected_cols).difference(df.columns)
    unexpected_cols = df.columns.difference(expected_cols)

    if len(missing_cols) > 0:
        fail_msgs.append(f"missing-cols(n={len(missing_cols)}) = {missing_cols}")
    if len(unexpected_cols) > 0:
        if unexpected_mode == "drop":
            if not fail_msgs:
                df = df.drop(columns=unexpected_cols)
        else:
            fail_msgs.append(f"unexpected-cols(n={len(unexpected_cols)}) = {unexpected_cols}")

    if fail_msgs:
        raise UnexpectedValueError(
            f"For {sport=} style={style.value} {features=} column validation failed! "
            + " ".join(fail_msgs)
        )

    return df[expected_cols]


def generate_train_test(
    df: pd.DataFrame,
    sport,
    style: DFSContestStyle,
    features: ModelFeatures,
    train_size: float = 0.25,
    random_state: None | int = None,
    service_as_feature: bool = False,
):
    """
    create regression train test data
    model_cols - if none then use all available columns
    return (X-train, X-test, y-top-train, y-top-test, y-last-win-train, y-last-win-test)
    """
    df = test_for_expected_cols(
        df, sport, style, unexpected_mode="drop", features=features, include_descriptive_cols=False
    )

    len_pre_na_drop = len(df)
    if not service_as_feature and "service" in df:
        df = df.drop(columns="service")
    df = df[df["top_winning_score"].notna()]
    df = df[df["last_winning_score"].notna()]
    if len(df) < len_pre_na_drop:
        _LOGGER.info(
            "Dropped %i rows of %i due to NaNs in top_score or last_winning_score",
            len_pre_na_drop - len(df),
            len_pre_na_drop,
        )
    if len(df) < 2:
        return None

    X = df.drop(columns=["top_winning_score", "last_winning_score"])
    y_top = df["top_winning_score"]
    y_last_win = df["last_winning_score"]

    try:
        sample_data = cast(
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series],
            train_test_split(
                X, y_top, y_last_win, random_state=random_state, train_size=train_size
            ),
        )
    except ValueError as ex:
        _LOGGER.info("generate_train_test_split:: Error generating train test split", exc_info=ex)
        return None

    return sample_data
