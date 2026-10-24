from datetime import date

import numpy as np

from odds_archive.compare import compare
from odds_archive.schema import to_archive_frame


def _game(game_date: date, **overrides) -> dict:
    row = {
        "date": game_date,
        "season": 2020,
        "away-team": "NYY",
        "away-abbr": "NYY",
        "home-team": "BOS",
        "home-abbr": "BOS",
        "away-moneyline": 120.0,
        "home-moneyline": -130.0,
        "overunder": 8.0,
        "over-odds": -110.0,
        "under-odds": -110.0,
        "spread": -1.5,
    }
    return {**row, **overrides}


_D1 = date(2020, 8, 1)
_D2 = date(2020, 8, 2)


def test_match_ignores_dtype_and_spread_sign():
    new_df = to_archive_frame([_game(_D1), _game(_D2, **{"over-odds": None, "under-odds": None})])
    ref_df = to_archive_frame([_game(_D1, spread=1.5), _game(_D2)])
    ref_df = ref_df.astype({"season": float})
    ref_df.loc[_D2, ["over-odds", "under-odds"]] = np.nan
    assert compare(new_df, ref_df).is_match


def test_missing_ref_column_not_compared():
    new_df = to_archive_frame([_game(_D1, **{"home-spread-odds": -110.0})])
    ref_df = to_archive_frame([_game(_D1)]).drop(columns=["away-spread-odds", "home-spread-odds"])
    comparison = compare(new_df, ref_df)
    assert comparison.missing_ref_cols == ["away-spread-odds", "home-spread-odds"]
    assert len(comparison.diffs_df) == 0
    assert not comparison.is_match


def test_differences():
    new_df = to_archive_frame([_game(_D1), _game(_D1, overunder=9.0), _game(_D2)])
    ref_df = to_archive_frame([_game(_D1, **{"home-moneyline": -140.0}), _game(_D1, overunder=9.5)])
    comparison = compare(new_df, ref_df)
    assert not comparison.is_match
    assert len(comparison.only_new_df) == 1 and len(comparison.only_ref_df) == 0
    assert comparison.diffs_df.column.to_list() == ["home-moneyline", "overunder"]
    assert comparison.diffs_df.new.to_list() == [-130.0, 9.0]
