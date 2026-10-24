"""
nflverse games export 'nflverse-games.csv'. One row per game with closing lines.
spread_line is positive when the home team is favored.
"""

import os

import pandas as pd
from fantasy_py import log

from ..schema import to_archive_frame
from .base import OddsSource, SourceContext

_LOGGER = log.get_logger(__name__)

_FILENAME = "nflverse-games.csv"
_SEASON_RANGE = (2009, 2025)
"""inclusive range of seasons used, others are skipped"""


class NflverseSource(OddsSource):
    NAME = "nflverse"
    SPORTS = frozenset(["nfl"])

    def load(self, ctx: SourceContext) -> pd.DataFrame:
        games_df = pd.read_csv(os.path.join(ctx.input_dir, _FILENAME))
        in_range = games_df.season.between(*_SEASON_RANGE)
        ctx.skipped["season out of range"] += int((~in_range).sum())
        games_df = games_df[in_range]

        game_rows = []
        for game in games_df.itertuples():
            game_date = pd.Timestamp(game.gameday).date()
            if (season := ctx.calendar.epoch_for_date(game_date).season) != game.season:
                raise ValueError(f"{game.game_id} on {game_date} is in {season=}")
            game_rows.append(
                {
                    "date": game_date,
                    "season": season,
                    "away-team": game.away_team,
                    "away-abbr": ctx.teams.abbr(game.away_team, season),
                    "home-team": game.home_team,
                    "home-abbr": ctx.teams.abbr(game.home_team, season),
                    "away-moneyline": game.away_moneyline,
                    "home-moneyline": game.home_moneyline,
                    "overunder": game.total_line,
                    "over-odds": game.over_odds,
                    "under-odds": game.under_odds,
                    "spread": -game.spread_line,
                }
            )
        _LOGGER.info("'%s' games=%i", _FILENAME, len(game_rows))
        return to_archive_frame(game_rows).sort_index(kind="stable")
