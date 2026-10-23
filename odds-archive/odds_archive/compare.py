"""Compare an archive to a reference archive by value (dtypes are ignored, spread compared as abs)"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fantasy_py import log

from .merge import GAME_KEY
from .schema import COLUMNS, INDEX_NAME

_LOGGER = log.get_logger(__name__)

_GAME_NUM = "game-num"
"""game number within a GAME_KEY, distinguishes doubleheaders"""
_STR_COLS = ["away-team", "home-team"]
_NUMERIC_COLS = [col for col in COLUMNS if col not in [*_STR_COLS, "away-abbr", "home-abbr"]]


@dataclass
class Comparison:
    only_new_df: pd.DataFrame
    only_ref_df: pd.DataFrame
    diffs_df: pd.DataFrame
    """one row per differing value with columns 'column', 'new', 'ref'"""

    @property
    def is_match(self) -> bool:
        return len(self.only_new_df) == len(self.only_ref_df) == len(self.diffs_df) == 0

    def log(self, max_rows: int = 20):
        if self.is_match:
            _LOGGER.success("archive matches reference")
            return
        with pd.option_context("display.max_rows", max_rows, "display.width", 200):
            _LOGGER.warning("%i games only in new archive:\n%s", len(self.only_new_df), self.only_new_df)
            _LOGGER.warning("%i games only in reference:\n%s", len(self.only_ref_df), self.only_ref_df)
            _LOGGER.warning(
                "%i value differences by column: %s\n%s",
                len(self.diffs_df),
                self.diffs_df.column.value_counts().to_dict(),
                self.diffs_df,
            )


def _keyed(archive_df: pd.DataFrame) -> pd.DataFrame:
    keyed_df = archive_df.reset_index()
    keyed_df[_GAME_NUM] = keyed_df.groupby(GAME_KEY).cumcount()
    keyed_df["spread"] = keyed_df.spread.abs()
    return keyed_df.set_index([*GAME_KEY, _GAME_NUM]).sort_index()


def _equal(new: pd.Series, ref: pd.Series, numeric: bool) -> pd.Series:
    both_null = new.isna() & ref.isna()
    if not numeric:
        return both_null | (new == ref)
    new_vals = pd.to_numeric(new).to_numpy(dtype=float)
    ref_vals = pd.to_numeric(ref).to_numpy(dtype=float)
    return both_null | pd.Series(np.isclose(new_vals, ref_vals), index=new.index)


def compare(new_df: pd.DataFrame, ref_df: pd.DataFrame) -> Comparison:
    new_keyed_df = _keyed(new_df)
    ref_keyed_df = _keyed(ref_df)
    common = new_keyed_df.index.intersection(ref_keyed_df.index)

    diff_dfs = []
    for col in [*_STR_COLS, *_NUMERIC_COLS]:
        new = new_keyed_df.loc[common, col]
        ref = ref_keyed_df.loc[common, col]
        differs = ~_equal(new, ref, col in _NUMERIC_COLS)
        if differs.any():
            diff_dfs.append(
                pd.DataFrame({"column": col, "new": new[differs], "ref": ref[differs]})
            )

    diffs_df = (
        pd.concat(diff_dfs).sort_index(level=INDEX_NAME, kind="stable")
        if len(diff_dfs) > 0
        else pd.DataFrame(columns=["column", "new", "ref"])
    )
    return Comparison(
        new_keyed_df.loc[new_keyed_df.index.difference(ref_keyed_df.index)],
        ref_keyed_df.loc[ref_keyed_df.index.difference(new_keyed_df.index)],
        diffs_df,
    )
