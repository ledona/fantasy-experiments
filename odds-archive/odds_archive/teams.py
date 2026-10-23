"""Map source team names to fantasy team abbreviations"""

import json
import os
from collections.abc import Callable

TEAM_ABBRS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "team_abbrs.json")


def _nba_season_rule(team_name: str, season: int) -> str | None:
    if team_name == "NewOrleans":
        # hornets up to 20122013, then the pelicans
        return "NOH" if season <= 20122013 else "NOP"
    if team_name == "Charlotte":
        # bobcats till 20132014, then hornets
        return "CHB" if season <= 20132014 else "CHA"
    return None


_SEASON_RULES: dict[str, Callable[[str, int], str | None]] = {"nba": _nba_season_rule}
"""sport -> func(team_name, season) returning an abbr for season dependent names, else None"""


class TeamAbbrs:
    """team name to abbreviation lookup for a sport, names without a mapping are returned unchanged"""

    def __init__(self, sport: str, remap: dict[str, str]):
        self.sport = sport
        self.remap = remap
        self._season_rule = _SEASON_RULES.get(sport)

    @classmethod
    def from_file(cls, sport: str, filepath: str = TEAM_ABBRS_PATH) -> "TeamAbbrs":
        with open(filepath, "r") as f_:
            return cls(sport, json.load(f_)[sport])

    def abbr(self, team_name: str, season: int) -> str:
        if self._season_rule is not None and (abbr := self._season_rule(team_name, season)):
            return abbr
        return self.remap.get(team_name, team_name)
