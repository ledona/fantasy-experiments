"""Combine source archives using source priority"""

import pandas as pd

from .schema import INDEX_NAME

GAME_KEY = [INDEX_NAME, "away-abbr", "home-abbr"]


def combine(source_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    source_dfs: source name -> archive df, in priority order (highest first)
    For each (date, away, home) key all rows come from the highest priority source with that key,
    so doubleheaders stay together.
    returns (combined df sorted by date, dropped lower priority rows with a 'source' column)
    """
    claimed: set[tuple] = set()
    kept_dfs = []
    dropped_dfs = []
    for name, source_df in source_dfs.items():
        keys = pd.Series(list(source_df.reset_index()[GAME_KEY].itertuples(index=False, name=None)))
        is_dup = keys.isin(claimed).to_numpy()
        kept_dfs.append(source_df[~is_dup])
        dropped_dfs.append(source_df[is_dup].assign(source=name))
        claimed.update(keys[~is_dup])

    combined_df = pd.concat(kept_dfs).sort_index(kind="stable")
    return combined_df, pd.concat(dropped_dfs)
