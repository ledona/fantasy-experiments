"""Registry of odds archive data sources"""

from .base import OddsSource, SourceContext
from .betiq import BetiqSource
from .sportsbook import SportsbookSource

SOURCES: dict[str, type[OddsSource]] = {
    SportsbookSource.NAME: SportsbookSource,
    BetiqSource.NAME: BetiqSource,
}
"""source name -> source class"""

SOURCE_PRIORITY: dict[str, list[str]] = {
    "mlb": ["sportsbook", "betiq"],
    "nba": ["sportsbook"],
    "nfl": ["sportsbook", "betiq"],
    "nhl": ["sportsbook"],
}
"""default sources by sport, highest priority first"""

__all__ = ["OddsSource", "SourceContext", "SOURCES", "SOURCE_PRIORITY"]
