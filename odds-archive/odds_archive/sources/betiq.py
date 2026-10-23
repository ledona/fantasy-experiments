"""
betiq TSV export '{SPORT}-odds.betiq.tsv'. One row per team per game, with spread and moneyline
from that team's perspective. Rows for a game are not necessarily consecutive.
"""

import os

import pandas as pd

from ..pairing import GameLines, pair_games
from ..schema import to_archive_frame
from .base import OddsSource, SourceContext

_LOCATION_VH = {"Away": "V", "Home": "H", "Neutral": "N"}
_GAME_KEY_COLS = ["date", "_team_1", "_team_2", "_team_1_score", "Total (O/U)"]
"""identifies a game's 2 rows, total distinguishes doubleheader games with the same score"""


def _betiq_lines(rows: pd.DataFrame, away_idx: int, home_idx: int) -> GameLines:
    ml = rows["Money Line"]
    return GameLines(
        ml.iloc[away_idx],
        ml.iloc[home_idx],
        rows.Spread.iloc[home_idx],
        rows["Total (O/U)"].iloc[home_idx],
    )


def game_ordered_rows(betiq_df: pd.DataFrame) -> pd.DataFrame:
    """add 'date' and 'VH' columns and order rows so that each game's 2 rows are consecutive"""
    if unknown := set(betiq_df.Location) - set(_LOCATION_VH):
        raise ValueError(f"Unknown Location values {unknown}")

    is_team_1 = betiq_df.Team <= betiq_df.Opponent
    reversed_score = betiq_df.Score.map(lambda score: "-".join(reversed(score.split("-"))))
    rows_df = betiq_df.assign(
        date=pd.to_datetime(betiq_df.Date).dt.date,
        VH=betiq_df.Location.map(_LOCATION_VH),
        _team_1=betiq_df.Team.where(is_team_1, betiq_df.Opponent),
        _team_2=betiq_df.Opponent.where(is_team_1, betiq_df.Team),
        _team_1_score=betiq_df.Score.where(is_team_1, reversed_score),
    ).sort_values(_GAME_KEY_COLS, kind="stable")

    game_sizes = rows_df.groupby(_GAME_KEY_COLS, dropna=False).size()
    if (game_sizes != 2).any():
        raise ValueError(f"Expected 2 rows per game, found:\n{game_sizes[game_sizes != 2]}")
    return rows_df


class BetiqSource(OddsSource):
    NAME = "betiq"
    SPORTS = frozenset(["mlb", "nfl"])

    def load(self, ctx: SourceContext) -> pd.DataFrame:
        filepath = os.path.join(ctx.input_dir, f"{ctx.sport}-odds.betiq.tsv")
        rows_df = game_ordered_rows(pd.read_csv(filepath, sep="\t"))
        game_rows = pair_games(
            rows_df, _betiq_lines, ctx.sport, ctx.calendar, ctx.teams, ctx.skipped
        )
        return to_archive_frame(game_rows).sort_index(kind="stable")
