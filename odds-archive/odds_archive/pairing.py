"""Convert consecutive away/home row pairs from a source into archive game rows"""

from collections import Counter
from collections.abc import Callable, Sequence
from typing import NamedTuple

import pandas as pd
from fantasy_py import NotSeasonDateError, log

from .seasons import SeasonCalendar
from .teams import TeamAbbrs

_LOGGER = log.get_logger(__name__)

_MAX_ABBR_LEN = 3
_NFL_SKIPPABLE_WEEKDAYS = (1, 2)
"""tues/wed, out of season nfl odds on these days are skipped"""


class SkipGame(Exception):
    """raised by a LinesFunc to skip a game, the message is the skip reason"""


class GameLines(NamedTuple):
    away_moneyline: float | None
    home_moneyline: float | None
    spread: float | None
    """home team spread, negative = home favored"""
    overunder: float | None
    over_odds: float | None = None
    under_odds: float | None = None
    away_spread_odds: float | None = None
    home_spread_odds: float | None = None


LinesFunc = Callable[[pd.DataFrame, int, int], GameLines]
"""func(game_rows_df, away_row_idx, home_row_idx) -> lines, raises SkipGame to skip the game"""


def _home_away_idx(rows: pd.DataFrame) -> tuple[int, int]:
    """for neutral site games (N/N) the first row is treated as away"""
    vh = rows.VH.to_list()
    if vh == ["V", "H"] or vh == ["N", "N"]:
        return 0, 1
    if vh == ["H", "V"]:
        return 1, 0
    raise ValueError(f"Expected VH values to be one of V/H, H/V or N/N. {vh=}")


def _game_row(
    rows: pd.DataFrame,
    lines_func: LinesFunc,
    sport: str,
    calendar: SeasonCalendar,
    teams: TeamAbbrs,
    required_cols: Sequence[str],
    skipped: Counter[str],
) -> dict | None:
    game_date = rows.date.iloc[0]
    away_idx, home_idx = _home_away_idx(rows)

    if any(col in rows and rows[col].hasnans for col in required_cols):
        skipped["missing " + "/".join(required_cols)] += 1
        return None
    if rows.date.nunique() > 1:
        _LOGGER.warning("Rows for game on %s have different dates:\n%s", game_date, rows)
    if "Rot" in rows and rows.Rot.iloc[0] + 1 != rows.Rot.iloc[1]:
        _LOGGER.warning("Rotation numbers on %s are not consecutive:\n%s", game_date, rows)

    try:
        epoch = calendar.epoch_for_date(game_date)
    except NotSeasonDateError:
        if sport == "nfl" and game_date.weekday() in _NFL_SKIPPABLE_WEEKDAYS:
            skipped["nfl tue/wed out of season"] += 1
            return None
        raise

    try:
        lines = lines_func(rows, away_idx, home_idx)
    except SkipGame as ex:
        skipped[str(ex)] += 1
        return None

    away_team = rows.Team.iloc[away_idx]
    home_team = rows.Team.iloc[home_idx]
    row = {
        "date": game_date,
        "season": epoch.season,
        "away-team": away_team,
        "away-abbr": teams.abbr(away_team, epoch.season),
        "home-team": home_team,
        "home-abbr": teams.abbr(home_team, epoch.season),
        "away-moneyline": lines.away_moneyline,
        "home-moneyline": lines.home_moneyline,
        "overunder": lines.overunder,
        "over-odds": lines.over_odds,
        "under-odds": lines.under_odds,
        "spread": lines.spread,
        "away-spread-odds": lines.away_spread_odds,
        "home-spread-odds": lines.home_spread_odds,
    }
    if len(row["away-abbr"]) > _MAX_ABBR_LEN or len(row["home-abbr"]) > _MAX_ABBR_LEN:
        raise ValueError(
            f"Team abbreviations on {game_date} are longer than {_MAX_ABBR_LEN}: "
            f"'{row['away-abbr']}', '{row['home-abbr']}'. Add a mapping to team_abbrs.json"
        )
    return row


def pair_games(
    rows_df: pd.DataFrame,
    lines_func: LinesFunc,
    sport: str,
    calendar: SeasonCalendar,
    teams: TeamAbbrs,
    skipped: Counter[str],
    required_cols: Sequence[str] = (),
) -> list[dict]:
    """
    rows_df: a consecutive row pair per game with 'date', 'VH' (V|H|N), 'Team', optional 'Rot'
        and the columns lines_func reads
    required_cols: games with a NaN in these are skipped
    returns game row dicts with keys 'date' + schema.COLUMNS
    """
    if len(rows_df) % 2 != 0:
        raise ValueError(f"Expected an even number of rows, found {len(rows_df)}")

    game_rows = []
    for i in range(0, len(rows_df), 2):
        rows = rows_df.iloc[i : i + 2]
        try:
            row = _game_row(rows, lines_func, sport, calendar, teams, required_cols, skipped)
        except Exception:
            _LOGGER.error("Failed to process game rows:\n%s", rows)
            raise
        if row is not None:
            game_rows.append(row)
    return game_rows
