import pandas as pd
import pytest

from odds_archive.teams import TeamAbbrs


def test_remap_and_passthrough():
    teams = TeamAbbrs("nhl", {"LosAngeles": "LAK"})
    assert teams.abbr("LosAngeles", 20202021) == "LAK"
    assert teams.abbr("BOS", 20202021) == "BOS"


@pytest.mark.parametrize(
    "team_name, season, expected",
    [
        ("NewOrleans", 20122013, "NOH"),
        ("NewOrleans", 20132014, "NOP"),
        ("Charlotte", 20132014, "CHB"),
        ("Charlotte", 20142015, "CHA"),
        ("Boston", 20132014, "BOS"),
    ],
)
def test_nba_season_rules(team_name, season, expected):
    assert TeamAbbrs("nba", {"Boston": "BOS"}).abbr(team_name, season) == expected


def test_season_rules_are_sport_specific():
    assert TeamAbbrs("nfl", {"NewOrleans": "NO"}).abbr("NewOrleans", 2012) == "NO"


@pytest.mark.parametrize("sport", ["mlb", "nba", "nfl", "nhl"])
def test_from_file(sport):
    teams = TeamAbbrs.from_file(sport)
    assert len(teams.remap) > 0
    assert {teams.abbr(abbr, 0) for abbr in teams.remap.values()} <= set(teams.names)


@pytest.mark.parametrize(
    "sport, team_name, expected",
    [("nfl", "JAX", "JAC"), ("nhl", "TB", "TBL"), ("mlb", "SFG", "SF"), ("mlb", "BRS", "BOS")],
)
def test_from_file_abbr_cleanup(sport, team_name, expected):
    """fantasy_py team_abbr_sub and source typo fixes"""
    assert TeamAbbrs.from_file(sport).abbr(team_name, 2020) == expected


def _archive_df(away: str, home: str) -> pd.DataFrame:
    return pd.DataFrame({"away-team": ["x"], "away-abbr": [away], "home-team": ["y"], "home-abbr": [home]})


def test_set_team_names():
    teams = TeamAbbrs("nfl", {}, {"KC": "Kansas City", "LA": "LA Rams"})
    df = teams.set_team_names(_archive_df("KC", "LA"))
    assert (df["away-team"].iloc[0], df["home-team"].iloc[0]) == ("Kansas City", "LA Rams")


def test_set_team_names_unknown_abbr_raises():
    teams = TeamAbbrs("nfl", {}, {"KC": "Kansas City"})
    with pytest.raises(ValueError, match="'XYZ'"):
        teams.set_team_names(_archive_df("KC", "XYZ"))
