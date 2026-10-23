from datetime import date

import pandas as pd
import pytest

from odds_archive.sources import SourceContext
from odds_archive.sources.betiq import BetiqSource
from odds_archive.teams import TeamAbbrs

_HEADER = "Date\tTeam\tOpponent\tLocation\tSpread\tTotal (O/U)\tMoney Line\tScore\n"

# rows for a game are not consecutive, last 4 rows are a doubleheader with the same score
_MLB_TSV = _HEADER + """\
2022-04-08\tBoston\tNY Yankees\tAway\t1.5\t8\t155\t5-6
2022-04-07\tMilwaukee\tChi Cubs\tAway\t-1.5\t9\t-175\t4-5
2022-04-08\tNY Yankees\tBoston\tHome\t-1.5\t8\t-170\t6-5
2022-04-07\tChi Cubs\tMilwaukee\tHome\t1.5\t9\t160\t5-4
2022-04-09\tToronto\tTexas\tAway\t1.5\t10\t156\t2-5
2022-04-09\tToronto\tTexas\tAway\t1.5\t9.5\t136\t2-5
2022-04-09\tTexas\tToronto\tHome\t-1.5\t9.5\t-144\t5-2
2022-04-09\tTexas\tToronto\tHome\t-1.5\t10\t-164\t5-2
"""

_TEAMS = {
    "Boston": "BOS",
    "NY Yankees": "NYY",
    "Milwaukee": "MIL",
    "Chi Cubs": "CHC",
    "Toronto": "TOR",
    "Texas": "TEX",
}


def _load(tmp_path, calendar_factory, tsv: str) -> pd.DataFrame:
    (tmp_path / "mlb-odds.betiq.tsv").write_text(tsv, encoding="utf-8")
    calendar = calendar_factory({2022: (date(2022, 4, 1), date(2022, 11, 5))})
    ctx = SourceContext("mlb", str(tmp_path), calendar, TeamAbbrs("mlb", _TEAMS))
    return BetiqSource().load(ctx)


def test_betiq(tmp_path, calendar_factory):
    df = _load(tmp_path, calendar_factory, _MLB_TSV)

    assert len(df) == 4
    games = {
        (row["away-abbr"], row["home-abbr"], row.overunder): row for _, row in df.iterrows()
    }
    game = games[("BOS", "NYY", 8)]
    assert (game["away-moneyline"], game["home-moneyline"], game.spread) == (155, -170, -1.5)
    assert game.season == 2022
    game = games[("MIL", "CHC", 9)]
    assert (game["away-moneyline"], game["home-moneyline"], game.spread) == (-175, 160, 1.5)

    # doubleheader games are paired by total
    assert games[("TOR", "TEX", 10)]["home-moneyline"] == -164
    assert games[("TOR", "TEX", 9.5)]["home-moneyline"] == -144
    assert list(df.index) == sorted(df.index)


def test_unpaired_game_raises(tmp_path, calendar_factory):
    tsv = _MLB_TSV.replace("-1.5\t9.5\t-144", "-1.5\t9\t-144")
    with pytest.raises(ValueError, match="2 rows per game"):
        _load(tmp_path, calendar_factory, tsv)


def test_unknown_location_raises(tmp_path, calendar_factory):
    with pytest.raises(ValueError, match="Location"):
        _load(tmp_path, calendar_factory, _MLB_TSV.replace("Away", "Road", 1))
