"""Season epoch lookups, wrapped in a protocol so tests can use a stub calendar"""

from datetime import date
from typing import Protocol, cast

from fantasy_py import SPORT_DB_MANAGER_DOMAIN, CLSRegistry, GameScheduleEpoch
from fantasy_py.sport import SportDBManager


class SeasonCalendar(Protocol):
    def epoch_for_date(self, date_: date) -> GameScheduleEpoch: ...

    def epoch_for_game_number(self, season: int, game_number: int) -> GameScheduleEpoch: ...


def sport_calendar(sport: str) -> SeasonCalendar:
    """the sport's db manager class, which provides epoch lookups without a database"""
    db_manager = cast(type[SportDBManager], CLSRegistry.get_class(SPORT_DB_MANAGER_DOMAIN, sport))
    return cast(SeasonCalendar, db_manager)
