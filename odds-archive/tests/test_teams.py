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
    assert len(TeamAbbrs.from_file(sport).remap) > 0
