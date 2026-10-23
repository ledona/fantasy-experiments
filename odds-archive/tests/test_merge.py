from datetime import date

from odds_archive.merge import combine
from odds_archive.schema import to_archive_frame


def _game(game_date: date, away: str, home: str, moneyline: float) -> dict:
    return {
        "date": game_date,
        "season": game_date.year,
        "away-team": away,
        "away-abbr": away,
        "home-team": home,
        "home-abbr": home,
        "away-moneyline": moneyline,
        "home-moneyline": -moneyline,
        "overunder": 8.0,
        "over-odds": None,
        "under-odds": None,
        "spread": -1.5,
    }


_D1 = date(2020, 8, 1)
_D2 = date(2020, 8, 2)


def test_no_overlap_is_concat_sorted():
    high_df = to_archive_frame([_game(_D2, "NYY", "BOS", 110)])
    low_df = to_archive_frame([_game(_D1, "NYY", "BOS", 120)])
    combined_df, dropped_df = combine({"high": high_df, "low": low_df})
    assert list(combined_df.index) == [_D1, _D2]
    assert len(dropped_df) == 0


def test_higher_priority_wins():
    high_df = to_archive_frame([_game(_D1, "NYY", "BOS", 110)])
    low_df = to_archive_frame([_game(_D1, "NYY", "BOS", 120), _game(_D2, "NYY", "BOS", 130)])
    combined_df, dropped_df = combine({"high": high_df, "low": low_df})
    assert combined_df["away-moneyline"].to_list() == [110, 130]
    assert dropped_df["away-moneyline"].to_list() == [120]
    assert dropped_df.source.to_list() == ["low"]


def test_doubleheader_kept_from_one_source():
    high_df = to_archive_frame([_game(_D1, "NYY", "BOS", 110), _game(_D1, "NYY", "BOS", 115)])
    low_df = to_archive_frame([_game(_D1, "NYY", "BOS", 120), _game(_D1, "NYY", "BOS", 125)])
    combined_df, dropped_df = combine({"high": high_df, "low": low_df})
    assert combined_df["away-moneyline"].to_list() == [110, 115]
    assert len(dropped_df) == 2
