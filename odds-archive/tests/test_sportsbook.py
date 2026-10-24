from collections import Counter
from datetime import date

import pandas as pd
import pytest

from odds_archive.sources import SourceContext, sportsbook
from odds_archive.sources.sportsbook import SportsbookSource, resolve_dates
from odds_archive.teams import TeamAbbrs

_MLB_CSV = """\
Date,Rot,VH,Team,Pitcher,1st,2nd,3rd,4th,5th,6th,7th,8th,9th,Final,Open,Close,Run Line,,Open OU,,Close OU,
405,997,V,STL,PA,1,1,0,0,1,0,0,0,0,3,-115,-102,1.5,-220,7.5,-105,7½,a-115
405,998,H,CUB,PB,0,0,0,0,0,0,0,0,0,0,-105,-108,-1.5,190,7.5,-115,7½,-105
406,901,V,COL,PC,4,0,2,4,0,0,0,0,0,10,140,138,1.5,-170,8,-115,NL,
406,902,H,ARI,PD,0,0,0,0,0,0,0,0,0,0,-150,-148,-1.5,150,8,-105,NL,
407,903,V,STL,PE,0,0,0,0,0,0,0,0,0,0,-150,,-1.5,150,8,-105,8,-110
407,904,H,ARI,PF,0,0,0,0,0,0,0,0,0,0,140,,1.5,-170,8,-115,8,-110
"""

_NHL_CSV = """\
NHL 2007-08,,,,,,,,,,,,,
Date,Rot,VH,Team,1st,2nd,3rd,Final,Open,Close,OpenOU,CloseOU,,
929,1,V,Anaheim,0,0,1,1,-155,-123,6,100,6,-105
929,2,H,LosAngeles,1,1,2,4,135,103,6,-120,6,-115
105,3,H,Anaheim,0,0,1,1,-140,-150,5.5,100,5.5,-110
105,4,V,LosAngeles,1,1,2,4,120,130,5.5,-120,5.5,-110
"""

_NBA_CSV = """\
NBA 2008-09,,,,,,,,,,,,
Date,Rot,VH,Team,1st,2nd,3rd,4th,Final,Open,Close,ML,2H
1028,501,V,Cleveland,28,22,13,22,85,179.5,182.5,220,93.5
1028,502,H,Boston,22,21,24,23,90,7,6,-260,5
1029,503,V,NewOrleans,28,22,13,22,85,200,PK,-110,93.5
1029,504,H,Boston,22,21,24,23,90,1,200.5,-110,5
1030,505,V,Cleveland,28,22,13,22,85,5,5,NL,93.5
1030,506,H,NewOrleans,22,21,24,23,90,210,210,NL,5
1031,507,V,Cleveland,28,22,13,22,85,5,5,,93.5
1031,508,H,Boston,22,21,24,23,90,210,210,-200,5
"""

_NFL_HEADER = "Date,Rot,VH,Team,1st,2nd,3rd,4th,Final,Open,Close,ML,2H\n"
_NFL_GAME = """\
{date},301,V,Carolina,0,10,0,0,10,40.5,40.5,121,24.5
{date},302,H,Denver,7,3,14,17,41,3,3,-141,3.5
"""

_TEAMS = {
    "mlb": {"CUB": "CHC"},
    "nba": {"Boston": "BOS", "Cleveland": "CLE"},
    "nfl": {"Carolina": "CAR", "Denver": "DEN"},
    "nhl": {"Anaheim": "ANA", "LosAngeles": "LAK"},
}


def _load(tmp_path, sport, calendar, files: dict[str, str]) -> tuple[pd.DataFrame, Counter]:
    for filename, contents in files.items():
        (tmp_path / filename).write_text(contents, encoding="utf-8")
    ctx = SourceContext(sport, str(tmp_path), calendar, TeamAbbrs(sport, _TEAMS[sport]))
    return SportsbookSource().load(ctx), ctx.skipped


@pytest.mark.parametrize(
    "mmdd, first_year, expected",
    [
        ([1001, 1231, 101, 615], 2007, [(2007, 10), (2007, 12), (2008, 1), (2008, 6)]),
        ([113, 707], 2021, [(2021, 1), (2021, 7)]),
        ([1022, 311, 730, 1011], 2019, [(2019, 10), (2020, 3), (2020, 7), (2020, 10)]),
        ([1008, 1007, 1009], 2014, [(2014, 10), (2014, 10), (2014, 10)]),
    ],
)
def test_resolve_dates(mmdd, first_year, expected):
    assert [(d.year, d.month) for d in resolve_dates(mmdd, first_year)] == expected


def test_resolve_dates_too_many_wraps():
    with pytest.raises(ValueError, match="2 calendar years"):
        resolve_dates([1201, 101, 1201, 101], 2010)


def test_mlb(tmp_path, calendar_factory):
    calendar = calendar_factory({2015: (date(2015, 4, 1), date(2015, 11, 5))})
    df, skipped = _load(tmp_path, "mlb", calendar, {"mlb-odds-2015.xlsx - Sheet1.csv": _MLB_CSV})

    assert skipped == {"missing Close": 1}
    assert len(df) == 2
    game = df.iloc[0]
    assert df.index[0] == date(2015, 4, 5)
    assert (game.season, game["away-abbr"], game["home-abbr"]) == (2015, "STL", "CHC")
    assert (game["away-moneyline"], game["home-moneyline"]) == (-102, -108)
    assert (game.overunder, game["over-odds"], game["under-odds"]) == (7.5, -115, -105)
    assert (game.spread, game["away-spread-odds"], game["home-spread-odds"]) == (-1.5, -220, 190)

    no_ou_game = df.iloc[1]
    assert pd.isna(no_ou_game.overunder)
    assert no_ou_game["over-odds"] is None and no_ou_game["under-odds"] is None
    assert (no_ou_game["home-moneyline"], no_ou_game.spread) == (-148, -1.5)


def test_mlb_run_line_used_over_moneyline(tmp_path, calendar_factory):
    """the run line underdog can be the moneyline favorite"""
    csv = _MLB_CSV.replace("-102,1.5,-220", "-102,-1.5,150").replace("-108,-1.5,190", "-108,1.5,-170")
    calendar = calendar_factory({2015: (date(2015, 4, 1), date(2015, 11, 5))})
    df, _ = _load(tmp_path, "mlb", calendar, {"mlb-odds-2015.xlsx - Sheet1.csv": csv})
    game = df.iloc[0]
    assert (game.spread, game["away-spread-odds"], game["home-spread-odds"]) == (1.5, 150, -170)


def test_mlb_run_line_not_opposite_is_null(tmp_path, calendar_factory):
    csv = _MLB_CSV.replace("-108,-1.5,190", "-108,1.5,190")
    calendar = calendar_factory({2015: (date(2015, 4, 1), date(2015, 11, 5))})
    df, _ = _load(tmp_path, "mlb", calendar, {"mlb-odds-2015.xlsx - Sheet1.csv": csv})
    game = df.iloc[0]
    assert pd.isna(game.spread)
    assert game["away-spread-odds"] is None and game["home-spread-odds"] is None


def test_mlb_no_run_line(tmp_path, calendar_factory):
    """files without a run line derive the spread from the moneyline and have no spread odds"""
    lines = [line.split(",") for line in _MLB_CSV.splitlines()]
    csv = "\n".join(",".join(cols[:17] + cols[19:]) for cols in lines) + "\n"
    calendar = calendar_factory({2010: (date(2010, 4, 1), date(2010, 11, 5))})
    df, _ = _load(tmp_path, "mlb", calendar, {"mlb-odds-2010.xlsx - Sheet1.csv": csv})
    game = df.iloc[0]
    assert (game.spread, game.overunder, game["over-odds"]) == (-1.5, 7.5, -115)
    assert game["away-spread-odds"] is None and game["home-spread-odds"] is None


def test_nhl(tmp_path, calendar_factory):
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2008, 6, 30))})
    df, _ = _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_CSV})

    assert list(df.index) == [date(2007, 9, 29), date(2008, 1, 5)]
    assert df.season.to_list() == [20072008, 20072008]
    game = df.iloc[0]
    assert (game["away-abbr"], game["away-moneyline"], game["home-moneyline"]) == ("ANA", -123, 103)
    assert (game.overunder, game["over-odds"], game["under-odds"]) == (6, -105, -115)
    assert game.spread == 1.5

    # home row first, home favored
    game = df.iloc[1]
    assert (game["away-abbr"], game["home-abbr"], game["home-moneyline"]) == ("LAK", "ANA", -150)
    assert game.spread == -1.5


@pytest.mark.parametrize(
    "away_ml, home_ml, expected_spread",
    [("-110", "-110", 1.5), ("0", "0", None), ("-20000", "150", None)],
)
def test_nhl_spread_from_moneyline(tmp_path, calendar_factory, away_ml, home_ml, expected_spread):
    """ties are not home favored, invalid moneylines give no spread"""
    csv = _NHL_CSV.replace("-155,-123,", f"-155,{away_ml},").replace("135,103,", f"135,{home_ml},")
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2008, 6, 30))})
    df, _ = _load(tmp_path, "nhl", calendar, {"Sheet1.csv": csv})
    spread = df.spread.iloc[0]
    assert pd.isna(spread) if expected_spread is None else spread == expected_spread


_NHL_PUCK_LINE_CSV = (
    _NHL_CSV.replace("Close,OpenOU,CloseOU,,", "Close,PuckLine,OpenOU,CloseOU,,,")
    .replace("-155,-123,", "-155,-123,1.5,-250,")
    .replace("135,103,", "135,103,-1.5,210,")
    .replace("-140,-150,", "-140,-150,-1.5,180,")
    .replace("120,130,", "120,130,1.5,-210,")
)


def test_nhl_puck_line(tmp_path, calendar_factory):
    """puck line is used over the moneyline favorite"""
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2008, 6, 30))})
    df, _ = _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_PUCK_LINE_CSV})
    away_ml_fav, home_first = df.iloc[0], df.iloc[1]
    assert (away_ml_fav.spread, away_ml_fav["away-spread-odds"], away_ml_fav["home-spread-odds"]) == (-1.5, -250, 210)
    assert (away_ml_fav.overunder, away_ml_fav["over-odds"]) == (6, -105)
    assert (home_first.spread, home_first["away-spread-odds"], home_first["home-spread-odds"]) == (-1.5, -210, 180)


def test_excluded_game_skipped(tmp_path, calendar_factory, monkeypatch):
    monkeypatch.setitem(sportsbook._EXCLUDED_GAMES, "nhl", {(date(2008, 1, 5), "LAK", "ANA")})
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2008, 6, 30))})
    df, skipped = _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_CSV})
    assert skipped == {"known bad source game": 1}
    assert list(df.index) == [date(2007, 9, 29)]


@pytest.mark.parametrize(
    "info, first_mmdd, seasons, expected_date, expected_season",
    [
        ("NHL 2012-13", 119, {20122013: (date(2013, 1, 15), date(2013, 6, 30))}, date(2013, 1, 19), 20122013),
        ("NHL 2021", 113, {20202021: (date(2021, 1, 1), date(2021, 7, 31))}, date(2021, 1, 13), 20202021),
    ],
)
def test_nhl_first_game_year(tmp_path, calendar_factory, info, first_mmdd, seasons, expected_date, expected_season):
    csv = _NHL_CSV.replace("NHL 2007-08", info).replace("929,", f"{first_mmdd},")
    csv = "\n".join(csv.splitlines()[:4]) + "\n"
    df, _ = _load(tmp_path, "nhl", calendar_factory(seasons), {"Sheet1.csv": csv})
    assert (df.index[0], df.season.iloc[0]) == (expected_date, expected_season)


def test_nba(tmp_path, calendar_factory):
    calendar = calendar_factory({20082009: (date(2008, 10, 1), date(2009, 6, 30))})
    df, skipped = _load(tmp_path, "nba", calendar, {"Sheet2.csv": _NBA_CSV})

    assert skipped == {"missing Close/ML": 1}
    assert len(df) == 3
    home_fav, pick, away_fav = (df.iloc[i] for i in range(3))
    assert (home_fav.spread, home_fav.overunder) == (-6, 182.5)
    assert (home_fav["away-moneyline"], home_fav["home-moneyline"]) == (220, -260)
    assert home_fav["over-odds"] is None
    assert (pick.spread, pick.overunder, pick["away-abbr"]) == (0, 200.5, "NOH")
    assert (away_fav.spread, away_fav.overunder) == (5, 210)
    assert pd.isna(away_fav["away-moneyline"]) and pd.isna(away_fav["home-moneyline"])


def test_nfl_skips(tmp_path, calendar_factory):
    """unsupported season files and out of season tue/wed games are skipped"""
    calendar = calendar_factory({2009: (date(2009, 9, 1), date(2010, 2, 15))}, [date(2009, 9, 15)])
    files = {
        "Sheet1.csv": "NFL 2007-08,,\n" + _NFL_HEADER + _NFL_GAME.format(date=906),
        "Sheet3.csv": "NFL 2009-10,,\n"
        + _NFL_HEADER
        + _NFL_GAME.format(date=910)
        + _NFL_GAME.format(date=915)
        + _NFL_GAME.format(date=103),
    }
    df, skipped = _load(tmp_path, "nfl", calendar, files)

    assert skipped == {"unsupported season file": 1, "nfl tue/wed out of season": 1}
    assert list(df.index) == [date(2009, 9, 10), date(2010, 1, 3)]
    assert df.spread.to_list() == [-3, -3]
    assert df.overunder.to_list() == [40.5, 40.5]


def test_duplicate_season_raises(tmp_path, calendar_factory):
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2008, 6, 30))})
    with pytest.raises(ValueError, match="Multiple files"):
        _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_CSV, "Sheet2.csv": _NHL_CSV})


def test_game_in_other_season_raises(tmp_path, calendar_factory):
    calendar = calendar_factory(
        {
            20072008: (date(2007, 9, 1), date(2007, 12, 31)),
            20082009: (date(2008, 1, 1), date(2008, 6, 30)),
        }
    )
    with pytest.raises(ValueError, match="not in file"):
        _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_CSV})


def test_out_of_season_game_raises(tmp_path, calendar_factory):
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2007, 12, 31))})
    with pytest.raises(ValueError, match="not in a season"):
        _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_CSV})


def test_bad_info_line_raises(tmp_path, calendar_factory):
    calendar = calendar_factory({20072008: (date(2007, 9, 1), date(2008, 6, 30))})
    with pytest.raises(ValueError, match="Expected first line"):
        _load(tmp_path, "nhl", calendar, {"Sheet1.csv": _NHL_CSV.replace("NHL", "NBA", 1)})
