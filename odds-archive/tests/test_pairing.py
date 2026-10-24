from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fantasy_py import NotSeasonDateError

from odds_archive.pairing import GameLines, SkipGame, pair_games
from odds_archive.teams import TeamAbbrs

_TEAMS = TeamAbbrs("nfl", {"Denver": "DEN", "Carolina": "CAR"})


def _lines(rows: pd.DataFrame, away_idx: int, home_idx: int) -> GameLines:
    return GameLines(rows.ML.iloc[away_idx], rows.ML.iloc[home_idx], -3.0, 40.5)


def _skip_lines(rows: pd.DataFrame, away_idx: int, home_idx: int) -> GameLines:
    raise SkipGame("test reason")


def _rows_df(vh=("V", "H"), game_date=date(2016, 9, 8), ml=(121, -141), team=("Carolina", "Denver")):
    return pd.DataFrame({"date": [game_date] * 2, "VH": list(vh), "Team": list(team), "ML": list(ml)})


@pytest.fixture
def calendar(calendar_factory):
    return calendar_factory({2016: (date(2016, 9, 1), date(2017, 2, 15))}, [date(2016, 9, 6)])


def _pair(rows_df, calendar, lines_func=_lines, required_cols=(), sport="nfl"):
    skipped: Counter[str] = Counter()
    rows = pair_games(rows_df, lines_func, sport, calendar, _TEAMS, skipped, required_cols)
    return rows, skipped


def test_away_home(calendar):
    rows, _ = _pair(_rows_df(), calendar)
    assert rows == [
        {
            "date": date(2016, 9, 8),
            "season": 2016,
            "away-team": "Carolina",
            "away-abbr": "CAR",
            "home-team": "Denver",
            "home-abbr": "DEN",
            "away-moneyline": 121,
            "home-moneyline": -141,
            "overunder": 40.5,
            "over-odds": None,
            "under-odds": None,
            "spread": -3.0,
            "away-spread-odds": None,
            "home-spread-odds": None,
        }
    ]


def test_home_away_order(calendar):
    rows, _ = _pair(_rows_df(vh=("H", "V"), team=("Denver", "Carolina"), ml=(-141, 121)), calendar)
    assert (rows[0]["away-abbr"], rows[0]["away-moneyline"]) == ("CAR", 121)
    assert (rows[0]["home-abbr"], rows[0]["home-moneyline"]) == ("DEN", -141)


def test_neutral_first_row_is_away(calendar):
    rows, _ = _pair(_rows_df(vh=("N", "N")), calendar)
    assert rows[0]["away-abbr"] == "CAR"


@pytest.mark.parametrize("vh", [("V", "V"), ("H", "N"), ("V", "N")])
def test_invalid_vh(calendar, vh):
    with pytest.raises(ValueError, match="VH"):
        _pair(_rows_df(vh=vh), calendar)


def test_odd_row_count(calendar):
    with pytest.raises(ValueError, match="even"):
        _pair(_rows_df().iloc[:1], calendar)


def test_required_nan_skipped(calendar):
    rows, skipped = _pair(_rows_df(ml=(np.nan, -141)), calendar, required_cols=["ML", "Close"])
    assert rows == []
    assert skipped == {"missing ML/Close": 1}


def test_lines_skip(calendar):
    rows, skipped = _pair(_rows_df(), calendar, lines_func=_skip_lines)
    assert rows == []
    assert skipped == {"test reason": 1}


def test_nfl_tue_wed_out_of_season_skipped(calendar):
    rows, skipped = _pair(_rows_df(game_date=date(2016, 9, 6)), calendar)
    assert rows == []
    assert skipped == {"nfl tue/wed out of season": 1}


@pytest.mark.parametrize(
    "sport, game_date",
    [("nfl", date(2016, 8, 1)), ("nba", date(2016, 9, 6))],
)
def test_out_of_season_raises(calendar, sport, game_date):
    """only nfl tue/wed dates are skipped"""
    with pytest.raises(NotSeasonDateError):
        _pair(_rows_df(game_date=game_date), calendar, sport=sport)


def test_long_abbr_raises(calendar):
    with pytest.raises(ValueError, match="team_abbrs.json"):
        _pair(_rows_df(team=("Carolina", "Unmapped")), calendar)
