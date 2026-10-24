"""
Sportsbook archive CSV files, one season per file.
mlb files have a header on the first line and the season in the filename ('mlb-odds-YYYY.xls...').
Other sports have a '{SPORT} YYYY[-YY]' line before the header.
Dates are MMDD and rows are in chronological order, two rows (away, home) per game.
"""

import glob
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd
from fantasy_py import NotSeasonDateError, log

from ..pairing import GameLines, SkipGame, pair_games
from ..schema import MONEYLINE_RANGE, to_archive_frame
from ..seasons import SeasonCalendar
from .base import OddsSource, SourceContext

_LOGGER = log.get_logger(__name__)

_MLB_FILENAME_RE = re.compile(r"mlb-odds-(\d{4})\.xls")
_YEAR_WRAP_MONTHS = 6
"""a month decrease larger than this between consecutive rows means the year rolled over"""
_OU_COLS = ["OpenOU", "OpenOU Odds", "CloseOU", "CloseOU Odds"]
"""names for the last 4 columns of mlb/nhl files, source headers for these are unreliable"""
_RUN_LINE_COLS = ["RunLine", "RunLine Odds"]
"""names for the 2 columns after Close in files with a run line (mlb) or puck line (nhl)"""
_RUN_LINE_NUM_COLS = {"mlb": 23, "nhl": 16}
"""column count of files with a run/puck line"""
_EXCLUDED_GAMES = {
    "nhl": {
        # 2014 cup final games 2-5 repeated at the end of the 2014-15 file
        (date(2015, 6, 7), "NYR", "LAK"),
        (date(2015, 6, 9), "LAK", "NYR"),
        (date(2015, 6, 11), "LAK", "NYR"),
        (date(2015, 6, 13), "NYR", "LAK"),
    },
}
"""sport -> (date, away abbr, home abbr) of bad source games"""
_NO_LINE = "NL"
_PICK = "pk"


@dataclass(frozen=True)
class _FileInfo:
    season: int
    first_year: int
    """calendar year of the first game in the file"""
    header_row: int


def _mmdd_date(year: int, mmdd: int) -> date:
    return date(year, mmdd // 100, mmdd % 100)


def resolve_dates(mmdd_vals: Sequence[int], first_year: int) -> list[date]:
    """convert chronological MMDD values to dates, the year increments when the month wraps"""
    year = first_year
    prev_month = mmdd_vals[0] // 100
    dates = []
    for mmdd in mmdd_vals:
        month = mmdd // 100
        if month < prev_month - _YEAR_WRAP_MONTHS:
            year += 1
        dates.append(_mmdd_date(year, mmdd))
        prev_month = month
    if year > first_year + 1:
        raise ValueError(f"Dates span more than 2 calendar years starting from {first_year}")
    return dates


def _inspect_file(filepath: str, sport: str, calendar: SeasonCalendar) -> _FileInfo:
    """raises NotSeasonDateError if the file's season is not supported"""
    if sport == "mlb":
        if (match := _MLB_FILENAME_RE.search(os.path.basename(filepath))) is None:
            raise ValueError(f"Expected mlb filename to match '{_MLB_FILENAME_RE.pattern}'")
        season = int(match[1])
        calendar.epoch_for_game_number(season, 1)
        return _FileInfo(season, season, 0)

    with open(filepath, "r") as f_:
        info_str = f_.readline().split(",", 1)[0]
        f_.readline()
        first_mmdd = int(f_.readline().split(",", 1)[0])

    prefix, _, years_str = info_str.partition(" ")
    if prefix.lower() != sport or not years_str.startswith("20"):
        raise ValueError(f"Expected first line to be '{sport.upper()} YYYY[-YY]'. {info_str=}")
    years = [int(year_str) for year_str in years_str.split("-")]
    min_year = years[0]

    if len(years) == 1:
        # single year file, season is that of the first game
        epoch = calendar.epoch_for_date(_mmdd_date(min_year, first_mmdd))
        return _FileInfo(epoch.season, min_year, 1)

    if len(years) != 2:
        raise ValueError(f"Expected at most 2 years. {info_str=}")
    max_year = years[1] + 2000 if years[1] < 100 else years[1]
    if max_year != min_year + 1:
        raise ValueError(f"Expected consecutive years. {info_str=}")
    season = min_year if sport == "nfl" else int(f"{min_year}{max_year}")
    calendar.epoch_for_game_number(season, 1)

    for year in (min_year, max_year):
        try:
            if calendar.epoch_for_date(_mmdd_date(year, first_mmdd)).season == season:
                return _FileInfo(season, year, 1)
        except NotSeasonDateError:
            pass
    raise ValueError(f"First game date {first_mmdd} is not in {season=}")


def _parse_odds(val) -> float | None:
    if val == _NO_LINE or pd.isna(val):
        return None
    if isinstance(val, str):
        # a few source values have an 'a' prefix
        val = val.removeprefix("a")
    return float(int(val))


def _parse_ou(val) -> float | None:
    if val == _NO_LINE or pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.replace("½", ".5")
    return float(val)


def _run_line(rows: pd.DataFrame, away_idx: int, home_idx: int) -> tuple[float | None, ...]:
    """returns (home spread, away odds, home odds), all None if away/home lines are not opposite"""
    run_line = [float(val) for val in rows.RunLine]
    if run_line[away_idx] != -run_line[home_idx]:
        return None, None, None
    odds = rows["RunLine Odds"]
    return run_line[home_idx], _parse_odds(odds.iloc[away_idx]), _parse_odds(odds.iloc[home_idx])


def _mlb_nhl_lines(rows: pd.DataFrame, away_idx: int, home_idx: int) -> GameLines:
    """
    moneyline is in Close. spread is the run/puck line if the file has one, else 1.5 by
    moneyline favorite (null for an invalid moneyline, +1.5 for a tie)
    """
    ml = [float(val) for val in rows.Close]
    if rows.CloseOU.nunique(dropna=False) != 1:
        raise ValueError(f"Expected both CloseOU values to match. {rows.CloseOU.to_list()=}")
    overunder = _parse_ou(rows.CloseOU.iloc[0])
    # over odds are on the first row, under odds on the second, regardless of V/H
    over_odds, under_odds = (
        (None, None)
        if overunder is None
        else (_parse_odds(rows["CloseOU Odds"].iloc[0]), _parse_odds(rows["CloseOU Odds"].iloc[1]))
    )
    if "RunLine" in rows:
        spread, away_spread_odds, home_spread_odds = _run_line(rows, away_idx, home_idx)
    else:
        valid_ml = all(MONEYLINE_RANGE[0] <= abs(val) <= MONEYLINE_RANGE[1] for val in ml)
        spread = None if not valid_ml else -1.5 if ml[home_idx] < ml[away_idx] else 1.5
        away_spread_odds = home_spread_odds = None
    return GameLines(
        ml[away_idx],
        ml[home_idx],
        spread,
        overunder,
        over_odds,
        under_odds,
        away_spread_odds,
        home_spread_odds,
    )


def _nba_nfl_lines(rows: pd.DataFrame, away_idx: int, home_idx: int) -> GameLines:
    """in Close the favorite's row has the spread and the other row has the overunder"""
    ml = [None, None] if rows.ML.isin([_NO_LINE]).any() else [float(val) for val in rows.ML]
    close = rows.Close.to_list()

    pick_idx = [i for i, val in enumerate(close) if isinstance(val, str) and val.lower() == _PICK]
    if len(pick_idx) > 0:
        if len(pick_idx) > 1:
            raise ValueError(f"Expected at most 1 pick'em Close value. {close=}")
        return GameLines(ml[away_idx], ml[home_idx], 0.0, float(close[1 - pick_idx[0]]))

    close = [float(val) for val in close]
    if close[away_idx] == close[home_idx]:
        raise ValueError(f"Close values should not be the same. {close=}")
    favorite_idx = away_idx if close[away_idx] < close[home_idx] else home_idx
    spread = close[favorite_idx]
    if spread < 0:
        _LOGGER.warning("Negative spread found, using abs:\n%s", rows)
        spread = abs(spread)
    return GameLines(
        ml[away_idx],
        ml[home_idx],
        -spread if favorite_idx == home_idx else spread,
        close[1 - favorite_idx],
    )


_LINES_FUNCS = {
    "mlb": (_mlb_nhl_lines, ["Close"]),
    "nhl": (_mlb_nhl_lines, ["Close"]),
    "nba": (_nba_nfl_lines, ["Close", "ML"]),
    "nfl": (_nba_nfl_lines, ["Close", "ML"]),
}
"""sport -> (lines func, required columns)"""


class SportsbookSource(OddsSource):
    NAME = "sportsbook"
    SPORTS = frozenset(_LINES_FUNCS)
    FILE_GLOB = "*Sheet*.csv"

    def load_file(self, filepath: str, ctx: SourceContext) -> tuple[int, pd.DataFrame] | None:
        """returns (season, archive df), or None if the file's season is not supported"""
        try:
            info = _inspect_file(filepath, ctx.sport, ctx.calendar)
        except NotSeasonDateError as ex:
            _LOGGER.warning("Skipping '%s', season is not supported: %s", filepath, ex)
            return None

        rows_df = pd.read_csv(filepath, header=info.header_row)
        if ctx.sport in ("mlb", "nhl"):
            rows_df.columns = [*rows_df.columns[:-4], *_OU_COLS]
        if ctx.sport in _RUN_LINE_NUM_COLS and len(rows_df.columns) == _RUN_LINE_NUM_COLS[ctx.sport]:
            close_idx = rows_df.columns.get_loc("Close")
            rows_df.columns = [
                *rows_df.columns[: close_idx + 1],
                *_RUN_LINE_COLS,
                *rows_df.columns[close_idx + 3 :],
            ]
        rows_df["date"] = resolve_dates(rows_df.Date.astype(int).to_list(), info.first_year)

        lines_func, required_cols = _LINES_FUNCS[ctx.sport]
        game_rows = pair_games(
            rows_df, lines_func, ctx.sport, ctx.calendar, ctx.teams, ctx.skipped, required_cols
        )
        file_df = to_archive_frame(game_rows)
        keys = zip(file_df.index, file_df["away-abbr"], file_df["home-abbr"])
        is_excluded = [key in _EXCLUDED_GAMES.get(ctx.sport, set()) for key in keys]
        if any(is_excluded):
            ctx.skipped["known bad source game"] += sum(is_excluded)
            file_df = file_df[[not excluded for excluded in is_excluded]]

        if (wrong_season := file_df.season != info.season).any():
            raise ValueError(
                f"Games in '{filepath}' are not in file {info.season=}:\n{file_df[wrong_season]}"
            )
        _LOGGER.info("'%s' season=%i games=%i", filepath, info.season, len(file_df))
        return info.season, file_df

    def load(self, ctx: SourceContext) -> pd.DataFrame:
        filepaths = sorted(glob.glob(os.path.join(ctx.input_dir, self.FILE_GLOB)))
        if len(filepaths) == 0:
            raise FileNotFoundError(f"No files matching '{self.FILE_GLOB}' in '{ctx.input_dir}'")

        season_files: dict[int, str] = {}
        file_dfs = []
        for filepath in filepaths:
            if (result := self.load_file(filepath, ctx)) is None:
                ctx.skipped["unsupported season file"] += 1
                continue
            season, file_df = result
            if season in season_files:
                raise ValueError(
                    f"Multiple files for {season=}: '{season_files[season]}', '{filepath}'"
                )
            season_files[season] = filepath
            file_dfs.append(file_df)
        return pd.concat(file_dfs).sort_index(kind="stable")
