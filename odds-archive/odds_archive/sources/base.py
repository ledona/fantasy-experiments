"""Base class for odds archive data sources"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from ..seasons import SeasonCalendar
from ..teams import TeamAbbrs


@dataclass
class SourceContext:
    sport: str
    input_dir: str
    """the sport's odds-archive data directory"""
    calendar: SeasonCalendar
    teams: TeamAbbrs
    skipped: Counter[str] = field(default_factory=Counter)
    """count of skipped games by reason, updated by the source"""


class OddsSource(ABC):
    """
    An odds data source. To add a source, subclass this, implement load and add the class
    to SOURCES and SOURCE_PRIORITY in sources/__init__.py
    """

    NAME: ClassVar[str]
    SPORTS: ClassVar[frozenset[str]]

    @abstractmethod
    def load(self, ctx: SourceContext) -> pd.DataFrame:
        """return all games from this source, as a dataframe created by schema.to_archive_frame"""
