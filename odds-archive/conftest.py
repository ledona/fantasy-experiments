"""Shared test fixtures, located here so that pytest adds this directory (and odds_archive) to sys.path"""

from collections.abc import Collection
from datetime import date
from types import SimpleNamespace

import pytest
from fantasy_py import NotSeasonDateError


class _StubCalendar:
    """SeasonCalendar with explicit season date ranges"""

    def __init__(self, seasons: dict[int, tuple[date, date]], excluded: Collection[date] = ()):
        self.seasons = seasons
        self.excluded = set(excluded)
        """dates inside a season range that are not in the season"""

    def epoch_for_date(self, date_: date):
        if date_ not in self.excluded:
            for season, (start, end) in self.seasons.items():
                if start <= date_ <= end:
                    return SimpleNamespace(season=season, date=date_)
        raise NotSeasonDateError(f"{date_} is not in a season")

    def epoch_for_game_number(self, season: int, game_number: int):
        if season not in self.seasons:
            raise NotSeasonDateError(f"{season=} is not supported")
        return SimpleNamespace(season=season, date=self.seasons[season][0])


@pytest.fixture
def calendar_factory():
    return _StubCalendar
