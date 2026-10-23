from datetime import date

import numpy as np
import pandas as pd
import pytest

from odds_archive.schema import COLUMNS, drop_no_odds, to_archive_frame, validate


def _row(**overrides) -> dict:
    row = {
        "date": date(2020, 1, 1),
        "season": 20192020,
        "away-team": "Away",
        "away-abbr": "AWY",
        "home-team": "Home",
        "home-abbr": "HOM",
        "away-moneyline": 120.0,
        "home-moneyline": -140.0,
        "overunder": 6.0,
        "over-odds": None,
        "under-odds": None,
        "spread": -1.5,
    }
    return {**row, **overrides}


def test_to_archive_frame_schema():
    df = to_archive_frame([_row(), _row(**{"over-odds": -110.0, "under-odds": -110.0})])
    assert df.index.name == "date"
    assert list(df.columns) == COLUMNS
    assert df.season.dtype == "int64"
    assert df["home-moneyline"].dtype == "float64"
    assert df["over-odds"].dtype == object
    assert df["over-odds"].iloc[0] is None
    assert df["over-odds"].iloc[1] == -110.0


def test_to_archive_frame_nan_odds_become_none():
    df = to_archive_frame([_row(**{"over-odds": np.nan})])
    assert df["over-odds"].iloc[0] is None


def test_validate_no_changes():
    df = to_archive_frame([_row(), _row(spread=1.5, **{"away-moneyline": -15000.0})])
    validated_df, nulled = validate(df, "nhl")
    pd.testing.assert_frame_equal(validated_df, df)
    assert sum(nulled.values()) == 0


@pytest.mark.parametrize(
    "col, bad_val",
    [
        ("away-moneyline", 20000.0),
        ("away-moneyline", -20000.0),
        ("away-moneyline", -50.0),
        ("spread", 2.5),
        ("spread", -2.5),
        ("overunder", 16.0),
    ],
)
def test_validate_nulls_only_invalid(col, bad_val):
    """valid negative values are kept when the column has an invalid value"""
    df = to_archive_frame([_row(), _row(**{col: bad_val})])
    validated_df, nulled = validate(df, "nhl")
    assert nulled[col] == 1
    assert pd.isna(validated_df[col].iloc[1])
    pd.testing.assert_frame_equal(validated_df.iloc[[0]], df.iloc[[0]])


def test_drop_no_odds():
    no_odds = {"away-moneyline": np.nan, "home-moneyline": np.nan, "overunder": np.nan, "spread": np.nan}
    df = to_archive_frame([_row(), _row(**no_odds), _row(**{**no_odds, "spread": 1.5})])
    dropped_df, count = drop_no_odds(df)
    assert count == 1
    assert dropped_df.spread.to_list() == [-1.5, 1.5]


def test_validate_nan_is_valid():
    df = to_archive_frame([_row(overunder=np.nan)])
    _, nulled = validate(df, "nhl")
    assert nulled["overunder"] == 0
