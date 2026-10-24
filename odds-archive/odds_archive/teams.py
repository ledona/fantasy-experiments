"""Map source team names to fantasy team abbreviations, and abbreviations to archive team names"""

import json
import os
from collections.abc import Callable

import pandas as pd

from .seasons import sport_db_manager

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEAM_ABBRS_PATH = os.path.join(_BASE_DIR, "team_abbrs.json")
TEAM_NAMES_PATH = os.path.join(_BASE_DIR, "team_names.json")
"""sport -> abbr -> archive team name, also defines the valid abbrs"""

_ABBR_FIXES = {"mlb": {"SFG": "SF", "BRS": "BOS"}}
"""abbr typos in source data"""


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

    def __init__(
        self,
        sport: str,
        remap: dict[str, str],
        names: dict[str, str] | None = None,
        abbr_sub: Callable[[str], str] | None = None,
    ):
        """
        names: abbr -> archive team name
        abbr_sub: func(abbr) -> true abbr, applied after remap
        """
        self.sport = sport
        self.remap = remap
        self.names = names
        self._abbr_sub = abbr_sub
        self._season_rule = _SEASON_RULES.get(sport)
        self._abbr_fixes = _ABBR_FIXES.get(sport, {})

    @classmethod
    def from_file(
        cls, sport: str, filepath: str = TEAM_ABBRS_PATH, names_filepath: str = TEAM_NAMES_PATH
    ) -> "TeamAbbrs":
        """uses the sport db manager's team_abbr_sub"""
        with open(filepath, "r") as f_:
            remap = json.load(f_)[sport]
        with open(names_filepath, "r") as f_:
            names = json.load(f_)[sport]
        return cls(sport, remap, names, sport_db_manager(sport).team_abbr_sub)

    def abbr(self, team_name: str, season: int) -> str:
        if self._season_rule is not None and (abbr := self._season_rule(team_name, season)):
            return abbr
        abbr = self.remap.get(team_name, team_name)
        abbr = self._abbr_fixes.get(abbr, abbr)
        return abbr if self._abbr_sub is None else self._abbr_sub(abbr)

    def set_team_names(self, archive_df: pd.DataFrame) -> pd.DataFrame:
        """set away-team/home-team from the abbrs, raises if an abbr has no name"""
        if self.names is None:
            raise ValueError("No team names defined")
        abbrs = set(archive_df["away-abbr"]) | set(archive_df["home-abbr"])
        if unknown := abbrs - set(self.names):
            raise ValueError(
                f"Unknown {self.sport} team abbrs {sorted(unknown)}. Add them to team_names.json "
                "or fix them with a team_abbrs.json mapping"
            )
        return archive_df.assign(
            **{
                "away-team": archive_df["away-abbr"].map(self.names),
                "home-team": archive_df["home-abbr"].map(self.names),
            }
        )
