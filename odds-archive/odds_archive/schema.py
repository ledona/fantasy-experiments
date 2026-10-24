"""
Odds archive schema and validation. One row per game, index 'date', american odds.
'spread' is the home spread (negative = home favored). Odds columns are None when unavailable.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

INDEX_NAME = "date"
COLUMNS = [
    "season",
    "away-team",
    "away-abbr",
    "home-team",
    "home-abbr",
    "away-moneyline",
    "home-moneyline",
    "overunder",
    "over-odds",
    "under-odds",
    "spread",
    "away-spread-odds",
    "home-spread-odds",
]
_FLOAT_COLS = ["away-moneyline", "home-moneyline", "overunder", "spread"]
_ODDS_LINE_COLS = {
    "over-odds": "overunder",
    "under-odds": "overunder",
    "away-spread-odds": "spread",
    "home-spread-odds": "spread",
}
"""odds column -> the line column it prices"""

VALID_RANGE: dict[str, dict[str, tuple[float, float]]] = {
    "mlb": {"spread": (1.5, 1.5), "overunder": (3, 18)},
    "nhl": {"spread": (1.5, 1.5), "overunder": (2.5, 15)},
    "nfl": {"spread": (0, 35), "overunder": (20, 85)},
    "nba": {"spread": (0, 25), "overunder": (150, 300)},
}
"""valid abs(value) range by sport and column"""
MONEYLINE_RANGE = (100, 15000)
"""valid abs(moneyline) range"""

SPORTS = tuple(VALID_RANGE)


def to_archive_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """create an archive schema dataframe from game row dicts (keys are 'date' + COLUMNS)"""
    df_ = pd.DataFrame(list(rows), columns=[INDEX_NAME, *COLUMNS]).set_index(INDEX_NAME)
    df_ = df_.astype({"season": "int64", **{col: "float64" for col in _FLOAT_COLS}})
    for col in _ODDS_LINE_COLS:
        df_[col] = _nan_to_none(df_[col])
    return df_


def _nan_to_none(vals: pd.Series) -> pd.Series:
    return vals.astype(object).where(vals.notna(), None)


def drop_no_odds(odds_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """drop games with no moneyline, overunder or spread, returns (df, dropped count)"""
    odds_cols = ["away-moneyline", "home-moneyline", "overunder", "spread"]
    has_odds = odds_df[odds_cols].notna().any(axis=1)
    return odds_df[has_odds], int((~has_odds).sum())


def validate(odds_df: pd.DataFrame, sport: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    null values whose abs is outside the valid range, and odds for a null line
    returns (validated df, nulled count by column)
    """
    # lines first, so odds for a nulled line are nulled
    ranges = {
        **VALID_RANGE[sport],
        "away-moneyline": MONEYLINE_RANGE,
        "home-moneyline": MONEYLINE_RANGE,
        **{col: MONEYLINE_RANGE for col in _ODDS_LINE_COLS},
    }
    nulled: dict[str, int] = {}
    validated_df = odds_df.copy()
    for col, (valid_min, valid_max) in ranges.items():
        abs_vals = pd.to_numeric(validated_df[col]).abs()
        invalid = abs_vals.notna() & ~abs_vals.between(valid_min, valid_max)
        if col in _ODDS_LINE_COLS:
            invalid |= abs_vals.notna() & validated_df[_ODDS_LINE_COLS[col]].isna()
        nulled[col] = int(invalid.sum())
        if nulled[col] > 0:
            validated_df[col] = validated_df[col].mask(invalid, np.nan)
    for col in _ODDS_LINE_COLS:
        validated_df[col] = _nan_to_none(validated_df[col])
    return validated_df, nulled
