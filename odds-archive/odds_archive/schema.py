"""
Odds archive output schema and range validation.

One row per game, index 'date' (datetime.date). Odds are american style.
'spread' is the home team's spread: negative = home favored, 0 = pick'em.
'over-odds'/'under-odds' are None when unavailable.
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
]
_FLOAT_COLS = ["away-moneyline", "home-moneyline", "overunder", "spread"]
_ODDS_COLS = ["over-odds", "under-odds"]

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
    for col in _ODDS_COLS:
        df_[col] = df_[col].astype(object).where(df_[col].notna(), None)
    return df_


def drop_no_odds(odds_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """drop games with no moneyline, overunder or spread, returns (df, dropped count)"""
    odds_cols = ["away-moneyline", "home-moneyline", "overunder", "spread"]
    has_odds = odds_df[odds_cols].notna().any(axis=1)
    return odds_df[has_odds], int((~has_odds).sum())


def validate(odds_df: pd.DataFrame, sport: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """null values whose abs is outside the valid range, returns (validated df, nulled count by column)"""
    ranges = {
        **VALID_RANGE[sport],
        "away-moneyline": MONEYLINE_RANGE,
        "home-moneyline": MONEYLINE_RANGE,
    }
    nulled: dict[str, int] = {}
    updates = {}
    for col, (valid_min, valid_max) in ranges.items():
        abs_vals = odds_df[col].abs()
        invalid = abs_vals.notna() & ~abs_vals.between(valid_min, valid_max)
        nulled[col] = int(invalid.sum())
        if nulled[col] > 0:
            updates[col] = odds_df[col].mask(invalid, np.nan)
    return odds_df.assign(**updates), nulled
