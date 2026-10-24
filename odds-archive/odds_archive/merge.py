"""Combine source archives using source priority"""

from collections import defaultdict
from datetime import date

import pandas as pd

from .schema import INDEX_NAME

GAME_KEY = [INDEX_NAME, "away-abbr", "home-abbr"]
DATE_TOLERANCE_DAYS = {"mlb": 0, "nba": 0, "nfl": 3, "nhl": 0}
"""max date difference for games with the same teams to be the same game"""

_Claimed = defaultdict[tuple[str, str], set[date]]
"""(away, home) -> game dates from higher priority sources"""


def _is_claimed(key: tuple, claimed: _Claimed, date_tolerance_days: int) -> bool:
    game_date, away, home = key
    return any(
        abs((game_date - claimed_date).days) <= date_tolerance_days
        for claimed_date in claimed[(away, home)]
    )


def combine(
    source_dfs: dict[str, pd.DataFrame], date_tolerance_days: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    source_dfs: source name -> archive df, highest priority first. A game is dropped if a
    higher priority source has the same away/home within date_tolerance_days.
    returns (combined df, dropped rows with a 'source' column)
    """
    claimed: _Claimed = defaultdict(set)
    kept_dfs = []
    dropped_dfs = []
    for name, source_df in source_dfs.items():
        keys = list(source_df.reset_index()[GAME_KEY].itertuples(index=False, name=None))
        is_dup = pd.Series([_is_claimed(key, claimed, date_tolerance_days) for key in keys], dtype=bool)
        kept_dfs.append(source_df[~is_dup.to_numpy()])
        dropped_dfs.append(source_df[is_dup.to_numpy()].assign(source=name))
        for game_date, away, home in (key for key, dup in zip(keys, is_dup) if not dup):
            claimed[(away, home)].add(game_date)

    combined_df = pd.concat(kept_dfs).sort_index(kind="stable")
    return combined_df, pd.concat(dropped_dfs)
