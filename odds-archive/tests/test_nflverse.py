from datetime import date

import pandas as pd
import pytest

from odds_archive.seasons import sport_db_manager
from odds_archive.sources import SourceContext
from odds_archive.sources.nflverse import NflverseSource
from odds_archive.teams import TeamAbbrs

_CSV = """\
game_id,season,gameday,away_team,home_team,away_moneyline,home_moneyline,spread_line,away_spread_odds,home_spread_odds,total_line,under_odds,over_odds
2008_01_KC_NE,2008,2008-09-07,KC,NE,,,-14,,,40,,
2009_01_JAX_IND,2009,2009-09-13,JAX,IND,250,-300,7,-108,-112,43.5,-105,-115
2009_02_GB_CIN,2009,2009-09-20,GB,CIN,-200,170,-4.5,,,42,-110,-110
2026_01_KC_NE,2026,2026-09-10,KC,NE,-150,130,-3,-110,-110,45,-110,-110
"""


def _load(tmp_path, calendar_factory, csv: str = _CSV) -> tuple[pd.DataFrame, dict]:
    (tmp_path / "nflverse-games.csv").write_text(csv, encoding="utf-8")
    calendar = calendar_factory({2009: (date(2009, 9, 1), date(2010, 2, 15))})
    teams = TeamAbbrs("nfl", {}, abbr_sub=sport_db_manager("nfl").team_abbr_sub)
    ctx = SourceContext("nfl", str(tmp_path), calendar, teams)
    return NflverseSource().load(ctx), ctx.skipped


def test_nflverse(tmp_path, calendar_factory):
    df, skipped = _load(tmp_path, calendar_factory)

    assert skipped == {"season out of range": 2}
    assert list(df.index) == [date(2009, 9, 13), date(2009, 9, 20)]
    home_fav, away_fav = df.iloc[0], df.iloc[1]
    assert (home_fav["away-abbr"], home_fav["home-abbr"], home_fav["away-team"]) == ("JAC", "IND", "JAX")
    assert (home_fav["away-moneyline"], home_fav["home-moneyline"]) == (250, -300)
    assert (home_fav.spread, home_fav.overunder) == (-7, 43.5)
    assert (home_fav["over-odds"], home_fav["under-odds"]) == (-115, -105)
    assert (home_fav["away-spread-odds"], home_fav["home-spread-odds"]) == (-108, -112)
    assert away_fav.spread == 4.5
    assert away_fav["away-spread-odds"] is None and away_fav["home-spread-odds"] is None


def test_wrong_season_raises(tmp_path, calendar_factory):
    with pytest.raises(ValueError, match="season="):
        _load(tmp_path, calendar_factory, _CSV.replace("2009_02_GB_CIN,2009", "2009_02_GB_CIN,2010"))
