import logging
import os
import re
from datetime import date, timedelta
from functools import partial
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    SPORT_DB_MANAGER_DOMAIN,
    CLSRegistry,
    DataNotAvailableException,
    DFSContestStyle,
    UnexpectedValueError,
    db,
)
from fantasy_py.betting import Contest, FiftyFifty, GeneralPrizePool
from sqlalchemy.orm import Session
from tqdm import tqdm

from ..modeling.generate_train_test import test_for_expected_cols
from .slate_scoring import (
    LOW_PLAYER_COST_PCTL,
    SlateScoreCacheMode,
    SlateScoreItem,
    get_stat_names,
    score_cache_ctx,
    slate_scoring,
)

_LOGGER = logging.getLogger(__name__)

_WARN_ON_SCREEN_LINEUP_CONSTRAINTS_ERR: set[tuple[str, str]] = {("mlb", "fanduel")}
""" 
set of tuples of (sport, service) that will only warn on a screen lineup 
constraint violation
"""

_SERVICE_NAME_TO_ABBR = {"fanduel": "fd", "draftkings": "dk", "yahoo": "y"}

# TODO: better solution for LOL would be to take the team name and use team_remaps
# remapping of team abbrs found in draft data to those found in database,
# dict[sport -> dict[service -> dict[draft abbr -> db abbr]]]
# if service is None, then use all services
_LOL_ABBR_REMAPS = {
    None: {
        "AF": "KF",  # Afreeca Freecs -> Kwangdong Freecs
        "AGO": "RGO",
        "APK": "SP",
        "EST": "ES",  # eStar
        "FCS": "S04",
        "FTC": "FNC",
        "FQ": "FLY",
        "IM": "IMT",
        "ITZ": "INTZ",
        "MG": "MSF",
        "ML": "MAD",
        "OHT": "100",
        "ROG": "RGE",
        "SB": "LSB",
        "SK": "SKG",  # SK Gaming
        "TD": "NS",
        "VFG": "GIA",
        "VG": "RA",
    },
    "fanduel": {
        "ES": "XL",  # Excel
    },
}


def _lol_abbr_remap(service, data_row) -> str:
    """for lol return a remapped team abbr if there is a remapping"""
    for service_key in [service, None]:
        if service_key in _LOL_ABBR_REMAPS and data_row.team_abbr in _LOL_ABBR_REMAPS[service_key]:
            return _LOL_ABBR_REMAPS[service_key][data_row.team_abbr]
    return data_row.team_abbr


def _nfl_abbr_remap(_, row):
    return (
        "OAK" if (row.team_abbr == "LV" and row.date < pd.Timestamp(2020, 7, 1)) else row.team_abbr
    )


def _mlb_abbr_remap(_, row):
    return (
        "WAS" if (row.team_abbr == "WSH" and row.date < pd.Timestamp(2020, 1, 1)) else row.team_abbr
    )


_ABBR_REMAPPERS: dict[str, Callable[[str, str], str]] = {
    "lol": _lol_abbr_remap,
    "nfl": _nfl_abbr_remap,
    "mlb": _mlb_abbr_remap,
}


def _get_nfl_showdown_features(session, slate_to_games: dict) -> pd.DataFrame:
    """Return per-slate NFL SHOWDOWN feature: td-to-yardage."""
    slate_game_values = ", ".join(
        f"({slate_id}, {game.id})" for slate_id, games in slate_to_games.items() for game in games
    )
    team_sql = f"""
    WITH slate_games(slate_id, game_id) AS (VALUES {slate_game_values})
    SELECT sg.slate_id, s.name AS stat_name, SUM(d.value) AS total
    FROM datum d
    JOIN slate_games sg ON sg.game_id = d.game_id
    JOIN statistic s ON d.statistic_id = s.id
    WHERE s.name IN ('yds', 'def_tds')
        AND d.player_id IS NULL
    GROUP BY sg.slate_id, s.name
    """
    player_sql = f"""
    WITH slate_games(slate_id, game_id) AS (VALUES {slate_game_values})
    SELECT sg.slate_id, s.name AS stat_name, SUM(d.value) AS total
    FROM datum d
    JOIN slate_games sg ON sg.game_id = d.game_id
    JOIN statistic s ON d.statistic_id = s.id
    WHERE s.name = 'tds'
        AND d.player_id IS NOT NULL
    GROUP BY sg.slate_id, s.name
    """
    conn = session.connection()
    pivoted = (
        pd.concat([pd.read_sql_query(team_sql, conn), pd.read_sql_query(player_sql, conn)])
        .pivot_table(index="slate_id", columns="stat_name", values="total", aggfunc="first")
        .rename_axis(None, axis=1)
        .fillna(0)
    )
    for col in ("yds", "def_tds", "tds"):
        if col not in pivoted.columns:
            pivoted[col] = 0.0

    result = pd.DataFrame(index=pivoted.index)
    result["td-to-yardage"] = (pivoted["tds"] + pivoted["def_tds"]) / pivoted["yds"]
    return result


def _showdown_team_stat_helper(
    session, slate_to_games: dict, stat_names: list[str]
) -> pd.DataFrame:
    """
    For SHOWDOWN slates (one game per slate), returns a dataframe indexed by slate_id
    with columns winning-team-{stat_name} and losing-team-{stat_name} for each requested stat.
    """
    game_id_to_slate = {games[0].id: slate_id for slate_id, games in slate_to_games.items()}
    game_ids_str = ", ".join(str(gid) for gid in game_id_to_slate)
    stat_names_str = ", ".join(f"'{s}'" for s in stat_names)
    sql = f"""
    SELECT d.game_id, d.team_id, s.name AS stat_name, d.value
    FROM datum d
    JOIN statistic s ON d.statistic_id = s.id
    WHERE s.name IN ({stat_names_str}) AND d.player_id IS NULL
        AND d.game_id IN ({game_ids_str})
    """
    df = pd.read_sql_query(sql, session.connection())
    df["slate_id"] = df["game_id"].map(game_id_to_slate)
    stat_dict = df.set_index(["slate_id", "team_id", "stat_name"])["value"].to_dict()

    rows = []
    for slate_id, games in slate_to_games.items():
        game = games[0]
        winner_id = game.winning_team_id
        assert winner_id, f"game {game.id} has no winner"
        loser_id = game.away_team_id if winner_id == game.home_team_id else game.home_team_id
        row = {"slate_id": slate_id}
        for stat_name in stat_names:
            row[f"winning-team-{stat_name}"] = stat_dict[(slate_id, winner_id, stat_name)]
            row[f"losing-team-{stat_name}"] = stat_dict[(slate_id, loser_id, stat_name)]
        rows.append(row)
    return pd.DataFrame(rows).set_index("slate_id")


def _create_team_score_df(
    session,
    slate_to_games: dict[int, list[db.Game]],
    percentile,
    style: DFSContestStyle,
    sport: str,
):
    """return a dataframe with final game score (e.g. mlb run total) features"""
    _LOGGER.info("calculating team scoring data for %d slates", len(slate_to_games))
    rows = [
        {
            "slate_id": slate_id,
            "game_id": game.id,
            "score_home": game.score_home,
            "score_away": game.score_away,
            "total_score": game.score_home + game.score_away,
            "high-team-score": max(game.score_home, game.score_away),
            "game-margin_of_victor": abs(game.score_home - game.score_away),
        }
        for slate_id, games in slate_to_games.items()
        for game in games
    ]
    db_team_score_df = pd.DataFrame(rows)
    if len(db_team_score_df) == 0:
        raise DataNotAvailableException(
            f"Empty team score df when retrieving for slates: {list(slate_to_games.keys())}"
        )

    melted_db_df = db_team_score_df.melt(
        id_vars=["slate_id", "game_id"], value_vars=["score_home", "score_away"]
    )
    if style.name == "CLASSIC":
        team_score_df = melted_db_df.groupby(["slate_id"]).agg(
            {"value": ["median", lambda x: np.percentile(x, percentile * 100), "sum"]}
        )
        team_score_df.columns = ["team_med", f"team-{percentile * 100}th_pctl", "total_score"]

        top3_sum = (
            db_team_score_df.groupby("slate_id")["total_score"]
            .apply(lambda x: x.nlargest(3).sum())
            .rename("top3-total")
        )
        team_score_df = team_score_df.join(top3_sum)
        if sport == "mlb":
            high_score = db_team_score_df.groupby("slate_id")["high-team-score"].apply(max)
            team_score_df = team_score_df.join(high_score)
    elif style.name == "SHOWDOWN":
        if db_team_score_df.game_id.duplicated().any():
            raise UnexpectedValueError("Showdown slate data should not have any duplicated games")
        cols = ["slate_id", "total_score", "game-margin_of_victor"]
        if sport == "mlb":
            cols.append("high-team-score")
        team_score_df = (
            db_team_score_df[cols]
            .set_index("slate_id")
            .assign(**{"game-total_score": db_team_score_df.total_score})
        )

        if sport == "nhl":
            nhl_showdown_stats_df = _showdown_team_stat_helper(session, slate_to_games, ["save"])
            team_score_df = team_score_df.join(nhl_showdown_stats_df).rename(
                columns={
                    "winning-team-save": "winning-team-saves",
                    "losing-team-save": "losing-team-saves",
                }
            )
        elif sport == "nfl":
            team_score_df = team_score_df.join(_get_nfl_showdown_features(session, slate_to_games))
        elif sport == "mlb":
            _LOGGER.info("Calculating mlb [winning|losing]-team-WH-allowed")
            mlb_showdown_stats_df = _showdown_team_stat_helper(
                session, slate_to_games, ["p_hits", "p_bb"]
            )
            wh_dict = {
                "winning-team-WH-allowed": (
                    mlb_showdown_stats_df["winning-team-p_bb"]
                    + mlb_showdown_stats_df["winning-team-p_hits"]
                ),
                "losing-team-WH-allowed": (
                    mlb_showdown_stats_df["losing-team-p_bb"]
                    + mlb_showdown_stats_df["losing-team-p_hits"]
                ),
            }
            mlb_showdown_stats_df = mlb_showdown_stats_df.assign(**wh_dict).drop(
                columns=[
                    "winning-team-p_bb",
                    "winning-team-p_hits",
                    "losing-team-p_bb",
                    "losing-team-p_hits",
                ]
            )
            team_score_df = team_score_df.join(mlb_showdown_stats_df)

    else:
        raise NotImplementedError(f"don't know how to do this for {style.name}")

    return team_score_df


def _get_exploded_pos_df(
    session,
    sport,
    slate_to_games: dict[int, list[db.Game]],
    cost_pos_drop: None | set,
    cost_pos_rename: None | dict,
    df_stat_names,
):
    slate_ids_str = ", ".join(str(sid) for sid in slate_to_games)
    # CTE maps each slate to its exact game IDs, fixing the doubleheader inaccuracy
    # that existed when games were inferred from date/season/team_id
    slate_game_values = ", ".join(
        f"({slate_id}, {game.id})" for slate_id, games in slate_to_games.items() for game in games
    )
    sql = f"""
    with slate_games(slate_id, game_id) as (values {slate_game_values})
    select dfc.daily_fantasy_slate_id as slate_id, dfc.positions as cost_positions,
        pp.abbr as stat_position,
        cd.value as score, dfc.team_id, dfc.player_id
    from daily_fantasy_cost dfc
        join slate_games sg on sg.slate_id = dfc.daily_fantasy_slate_id
        join calculation_datum cd on (
            cd.game_id = sg.game_id and
            cd.player_id is dfc.player_id and
            cd.team_id = dfc.team_id
        )
        join statistic s on cd.statistic_id = s.id
        join player p on dfc.player_id = p.id
        join player_position pp on p.player_position_id = pp.id
    where dfc.daily_fantasy_slate_id in ({slate_ids_str}) and
        s.name in ({df_stat_names})
    """
    db_df = pd.read_sql_query(sql, session.connection())

    if len(db_df) == 0:
        raise DataNotAvailableException("No exploded positional data returned!")

    db_manager = CLSRegistry.get_class(SPORT_DB_MANAGER_DOMAIN, sport)

    def apply_func(row):
        """
        use cost positions if available and valid
        otherwise try to use stat_pos_to_cost_pos if available
        otherwise use stat_position
        """
        if row.cost_positions is not None and "UNKNOWN" not in row.cost_positions.upper():
            return row.cost_positions
        if db_manager.STAT_POSITION_TO_COST_POSITIONS is not None:
            cost_positions = db_manager.STAT_POSITION_TO_COST_POSITIONS.get(row.stat_position)
            return "/".join(cost_positions) if cost_positions else row.stat_position
        return row.stat_position

    db_df["position"] = db_df.apply(apply_func, axis=1)

    db_exploded_pos_df = db_df.assign(position=db_df.position.str.split("/")).explode("position")

    if cost_pos_drop is not None:
        db_exploded_pos_df = db_exploded_pos_df.query("position not in @cost_pos_drop")
    if cost_pos_rename is not None:
        for old_pos, new_pos in cost_pos_rename.items():
            db_exploded_pos_df.loc[db_exploded_pos_df.position == old_pos, "position"] = new_pos
    return db_exploded_pos_df


_DK_WARNED_GAME_TYPE_PATTERNS = [
    r"Best Ball",
    r"Campbell.s Chunky. Pick",
    r"Madden Classic",
    r"Series",
    r"Single Stat.*",
    r"Starting 5",
    r"WNBA",
]
"""uncategorized draftkings game types to warn about"""


def _infer_contest_style(service, row) -> DFSContestStyle:
    """get contest data"""
    if service == "draftkings":
        if row.dk_game_type == "Classic":
            return DFSContestStyle.CLASSIC
        if "Showdown" in row.dk_game_type:
            return DFSContestStyle.SHOWDOWN
        if "Tiers" in row.dk_game_type:
            return DFSContestStyle.DK_TIERS
        for warn_pattern in _DK_WARNED_GAME_TYPE_PATTERNS:
            if re.match(warn_pattern, row.dk_game_type):
                _LOGGER.limited_warning(
                    "Unhandled dk game type found. '%s'",
                    row.dk_game_type,
                    limit=1,
                    limit_show_hidden_warning=False,
                )
                return row.dk_game_type
        raise UnexpectedValueError(f"Unexpected dk game type {row.dk_game_type=}")

    if service == "fanduel":
        if "@" in (row.title or ""):
            return DFSContestStyle.SHOWDOWN
        return DFSContestStyle.CLASSIC

    if service == "yahoo":
        if (
            " Cup " in row.title
            or " to 1st]" in row.title
            or " 50/50" in row.title
            or "QuickMatch vs " in row.title
            or "H2H vs " in row.title
            or "-Team" in row.title
            or "Freeroll" in row.title  # N-team contests are classic
            or "Quadruple Up" in row.title
            or "Guaranteed" in row.title
        ):
            return DFSContestStyle.CLASSIC

    raise NotImplementedError(f"Could not infer contest style for {service=} {row.title=}")


def _infer_contest_type(service, title) -> str:
    if service == "draftkings":
        if re.match(r".* vs\. [^)]+$", title):
            return "H2H"
        return FiftyFifty.TYPE_NAME if "Double Up" in title else GeneralPrizePool.TYPE_NAME
    if service == "fanduel":
        if "Head-to-head" in (title or ""):
            return "H2H"
        if (title or "").startswith("50/50"):
            return FiftyFifty.TYPE_NAME
        return GeneralPrizePool.TYPE_NAME
    if service == "yahoo":
        if " QuickMatch vs " in title or "H2H vs " in title:
            return "H2H"
        if " 50/50" in title:
            return FiftyFifty.TYPE_NAME
        if (
            " Cup " in title
            or " to 1st]" in title
            or "Freeroll" in title
            or "Quadruple Up" in title
            or
            # multi-team games are GPP if not caught by 50/50
            "-Team" in title
            or
            # treat winner takes all like a gpp
            title.endswith("Team Winner Takes All")
            or "Guaranteed" in title
        ):
            return GeneralPrizePool.TYPE_NAME
    raise NotImplementedError(f"Could not infer contest type for {service=} {title=}")


def _infer_contest_info_cols(service, row):
    return {
        "type": _infer_contest_type(service, row.title),
        "style": _infer_contest_style(service, row),
    }


_EXPECTED_CONTEST_COLS = {
    "contest_id",
    "date",
    "title",
    "top_score",
    "last_winning_score",
    "entries",
}


def _get_contest_df(
    service_name, sport, style, contest_type: Contest, min_date, max_date, contest_data_path
) -> pd.DataFrame:
    """
    create a dataframe from the contest dataset
    """
    contest_csv_path = os.path.join(contest_data_path, service_name + ".contest.csv")
    contest_df = pd.read_csv(contest_csv_path, parse_dates=["date"]).query(
        "sport == @sport and @min_date <= date < @max_date"
    )
    if len(contest_df) == 0:
        raise DataNotAvailableException(
            f"No contest data for {sport=} {min_date=} {max_date=} found in '{contest_csv_path}'"
        )
    if missing_cols := _EXPECTED_CONTEST_COLS.difference(contest_df.keys()):
        raise DataNotAvailableException(
            f"Not all expected contest data columns were found in '{contest_csv_path}'. Missing cols={missing_cols}"
        )
    contest_df.date = contest_df.date.dt.normalize()
    contest_df = contest_df.where(contest_df.notnull(), None)

    # add style and type
    contest_type_cols = contest_df.apply(
        partial(_infer_contest_info_cols, service_name), axis=1, result_type="expand"
    )
    contest_df = pd.concat([contest_df, contest_type_cols], axis=1)
    queries = []
    if style is not None:
        queries.append("style == @style")
    if contest_type is not None:
        queries.append(f"type == '{contest_type.TYPE_NAME}'")
    if len(queries) > 0:
        contest_df = contest_df.query(" and ".join(queries))

    betting_csv_path = os.path.join(contest_data_path, service_name + ".betting.csv")
    bet_df = (
        pd.read_csv(betting_csv_path)
        .drop_duplicates("contest_id")
        .set_index("contest_id")[["link"]]
    )
    contest_df = contest_df.merge(bet_df, how="left", on="contest_id")
    return contest_df


def _get_draft_df(service, sport, style, min_date, max_date, contest_data_path) -> pd.DataFrame:
    """load drafted players from scraped dataset"""
    csv_path = os.path.join(contest_data_path, service + ".draft.csv")
    draft_df = pd.read_csv(csv_path, parse_dates=["date"]).query(
        "sport == @sport and @min_date <= date < @max_date"
    )
    if len(draft_df) == 0:
        raise DataNotAvailableException(
            f"No draft data found for {sport=}, {service=}, {style=}, {min_date=}, "
            f"{max_date=} in '{csv_path}'. Perhaps the last data retrieval run had "
            "too many constraints (date for example)?"
        )

    draft_df["service"] = draft_df.contest.map(lambda contest: contest.split("-", 1)[0])
    draft_df.team_abbr = draft_df.team_abbr.str.upper()
    if sport in _ABBR_REMAPPERS:
        draft_df.team_abbr = draft_df.apply(
            partial(_ABBR_REMAPPERS[sport], service),
            axis=1,
        )
    service_abbr = _SERVICE_NAME_TO_ABBR[service]
    draft_df = draft_df.query("service == @service_abbr and team_abbr.notnull()")[
        ["position", "name", "team_abbr", "contest_id"]
    ]

    return draft_df


def _create_teams_contest_df(contest_df, draft_df, service, sport):
    """group contests together and create team sets used in each contest"""
    service_cls = CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, service)
    abbr_remaps = service_cls.TEAM_ABBR_ALIASES.get(sport)

    # add team/lineup draft data
    contest_draft_df = pd.merge(contest_df, draft_df, on="contest_id")
    contest_draft_df["team_abbr"] = contest_draft_df.team_abbr.map(
        lambda abbr: (abbr_remaps.get(abbr) or abbr) if abbr_remaps else abbr
    )

    def _common_title(title_series: pd.Series) -> str:
        """the title of a contest will be the common prefix amongst all the possible contest titles"""
        title_list = title_series.tolist()
        return "" if None in title_list else os.path.commonprefix(title_list)

    tc_df = pd.DataFrame(
        contest_draft_df.groupby(["contest_id", "date", "style", "type", "link", "entries"]).agg(
            {
                "team_abbr": set,
                "title": _common_title,
                "top_score": lambda score: score.mean(),
                "last_winning_score": lambda score: score.mean(),
            }
        )
    ).reset_index()
    tc_df = tc_df.rename(columns={"team_abbr": "teams"})
    tc_df["draft_team_count"] = tc_df.teams.map(len)
    return tc_df


def _get_slate_df(session: Session, service, style, min_date, max_date):
    sql = f"""
    select distinct daily_fantasy_slate.id as slate_id, date, 
        daily_fantasy_slate.name as slate_name, style as contest_style, abbr
    from daily_fantasy_slate 
        join daily_fantasy_cost on daily_fantasy_slate.id = daily_fantasy_cost.daily_fantasy_slate_id
        join team on team_id = team.id
    where service = '{service}' and date between '{min_date}' and date('{max_date}', '-1 days')
    """
    if style is not None:
        sql += f" and style = '{style.name}'"

    db_df = pd.read_sql_query(sql, session.connection(), parse_dates=["date"])
    if len(db_df) == 0:
        raise DataNotAvailableException("No slates found for", service, style, min_date, max_date)

    # get team sets
    slate_db_df = pd.DataFrame(
        db_df.groupby(["slate_id", "date", "slate_name", "contest_style"]).agg({"abbr": set})
    ).reset_index()

    try:
        slate_db_df = slate_db_df.set_index("date").rename(columns={"abbr": "teams"})
    except Exception as ex:
        raise ValueError("Error processing slate db df", slate_db_df) from ex

    if style != DFSContestStyle.SHOWDOWN:
        slate_db_df["team_count"] = slate_db_df.teams.map(len)
    return slate_db_df


_NO_SLATE_ID_FOUND = pd.Series({"slate_id": None})


def _get_slate_id(contest_row: pd.Series, slate_db_df: pd.DataFrame) -> pd.Series:
    """
    guesses the db slate id for the contest_row
    returns - series of (slate_id, number of teams playing in slate)
    """
    try:
        date_slates = slate_db_df.loc[[contest_row.date]]
    except KeyError:
        _LOGGER.warning(
            "get_slate_id::Key error/No slates found for %s %s '%s'",
            contest_row.date,
            contest_row.style,
            contest_row.title,
        )
        return _NO_SLATE_ID_FOUND
    if "team_count" in date_slates:
        date_slates = date_slates.sort_values("team_count")
    try:
        slates = date_slates.query(
            "@contest_row.teams <= teams and contest_style == @contest_row.style.name"
        )
    except Exception as err:
        _LOGGER.warning(
            "get_slate_id:: Unhandled exception querying for teams date %s",
            contest_row.date,
            exc_info=err,
        )
        return _NO_SLATE_ID_FOUND

    slates_found = len(slates)
    if slates_found == 0:
        _LOGGER.warning(
            "get_slate_id::No slates found for %s that matches teams %s.",
            contest_row.date,
            contest_row.teams,
        )
        return _NO_SLATE_ID_FOUND

    cols = ["slate_id"]
    if "team_count" in slates:
        cols.append("team_count")
    return slates.iloc[0][cols]


def _get_nhl_3player_line_goal_pct(session, slate_to_games: dict) -> pd.Series:
    """
    Calculate 3-player-line-goals% per slate, indexed by slate_id.

    infer 3 player line goals for a line by using for each line
    the minimum of (goals, assists/2)
    sum this across all lines then divide by total goals
    """
    slate_game_values = ", ".join(
        f"({slate_id}, {game.id})" for slate_id, games in slate_to_games.items() for game in games
    )
    total_goals_sql = f"""
    WITH slate_games(slate_id, game_id) AS (VALUES {slate_game_values})
    SELECT sg.slate_id, SUM(d.value) AS total_goals
    FROM datum d
    JOIN slate_games sg ON sg.game_id = d.game_id
    JOIN statistic s ON d.statistic_id = s.id
    WHERE s.name = 'goal' AND d.player_id IS NULL
    GROUP BY sg.slate_id
    """
    fwd_stats_sql = f"""
    WITH slate_games(slate_id, game_id) AS (VALUES {slate_game_values})
    SELECT sg.slate_id, d.game_id, d.team_id, d.player_id, s.name AS stat_name, d.value
    FROM datum d
    JOIN slate_games sg ON sg.game_id = d.game_id
    JOIN statistic s ON d.statistic_id = s.id
    JOIN player p ON d.player_id = p.id
    JOIN player_position pp ON p.player_position_id = pp.id
    WHERE s.name IN ('goal', 'assist', 'line')
        AND d.player_id IS NOT NULL
        AND pp.abbr NOT IN ('G', 'D')
    """
    conn = session.connection()
    total_goals = pd.read_sql_query(total_goals_sql, conn).set_index("slate_id")["total_goals"]
    fwd_df = pd.read_sql_query(fwd_stats_sql, conn)

    if fwd_df.empty:
        raise DataNotAvailableException("No NHL forward stat data for 3-player-line-goals%")

    pivoted = (
        fwd_df.pivot_table(
            index=["slate_id", "game_id", "team_id", "player_id"],
            columns="stat_name",
            values="value",
            aggfunc="first",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for col in ("goal", "assist", "line"):
        if col not in pivoted.columns:
            pivoted[col] = np.nan

    pivoted["goal"] = pivoted["goal"].fillna(0)
    pivoted["assist"] = pivoted["assist"].fillna(0)

    fwd_with_line = pivoted[pivoted["line"].notna()]
    line_groups = (
        fwd_with_line.groupby(["slate_id", "game_id", "team_id", "line"])
        .agg(line_goals=("goal", "sum"), line_assists=("assist", "sum"))
        .reset_index()
    )
    line_groups["inferred_line_goals"] = np.minimum(
        line_groups["line_goals"], line_groups["line_assists"] / 2
    )
    inferred_goals = line_groups.groupby("slate_id")["inferred_line_goals"].sum()
    lgp = (inferred_goals / total_goals).rename("3-player-line-goals%")
    assert len(slate_to_games) == len(lgp)
    return lgp


def _get_nba_low_cost_high_use(session, slate_to_games: dict) -> pd.Series:
    """Count of players per slate who are low cost (below 25th pctl) and played 25+ minutes."""
    slate_ids_str = ", ".join(str(sid) for sid in slate_to_games)
    slate_game_values = ", ".join(
        f"({slate_id}, {game.id})" for slate_id, games in slate_to_games.items() for game in games
    )
    cost_sql = f"""
    SELECT daily_fantasy_slate_id AS slate_id, player_id, cost
    FROM daily_fantasy_cost
    WHERE daily_fantasy_slate_id IN ({slate_ids_str}) AND player_id IS NOT NULL
    """
    time_sql = f"""
    WITH slate_games(slate_id, game_id) AS (VALUES {slate_game_values})
    SELECT sg.slate_id, d.player_id, SUM(d.value) AS seconds_played
    FROM datum d
    JOIN slate_games sg ON sg.game_id = d.game_id
    JOIN statistic s ON d.statistic_id = s.id
    WHERE s.name = 'time' AND d.player_id IS NOT NULL
    GROUP BY sg.slate_id, d.player_id
    """
    conn = session.connection()
    cost_df = pd.read_sql_query(cost_sql, conn)
    time_df = pd.read_sql_query(time_sql, conn)
    merged = cost_df.merge(time_df, on=["slate_id", "player_id"], how="inner")
    cost_threshold = merged.groupby("slate_id")["cost"].transform(
        lambda x: np.percentile(x, LOW_PLAYER_COST_PCTL * 100)
    )
    qualified_players_df = merged[
        (merged["cost"] < cost_threshold) & (merged["seconds_played"] >= 1500)
    ]
    lchu = qualified_players_df.groupby("slate_id").size().rename("low_cost_high_use")
    for slate_id in set(slate_to_games.keys()).difference(lchu.index):
        lchu[slate_id] = 0
    return lchu


def _get_player_scores(session, cfg, sport, style, slate_to_games, top_percentile, df_stat_names):
    """
    return a dataframe containing slate player features
    - the top_percentileth player dfs score for
      each position for the requested slate(s), sport and service
    """
    _LOGGER.info("calculating player scores for %d slates", len(slate_to_games))

    # TODO: exploded scores is only needed for legacy. if the new feature set is better then drop exploded score retrieval
    db_exploded_pos_df = _get_exploded_pos_df(
        session,
        sport,
        slate_to_games,
        cfg.get("cost_pos_drop"),
        cfg.get("cost_pos_rename"),
        df_stat_names,
    )

    player_scores_df = (
        db_exploded_pos_df[["slate_id", "position", "score"]]
        .groupby(["slate_id", "position"])
        .agg(["median", lambda x: np.percentile(x, top_percentile * 100)])
    )
    player_scores_df.columns = ["med-dfs", f"{top_percentile * 100}th-pctl-dfs"]
    player_scores_df = player_scores_df.reset_index(level="position").pivot(
        columns="position",
        values=["med-dfs", f"{top_percentile * 100}th-pctl-dfs"],
    )
    player_scores_df.columns = player_scores_df.columns.map(
        lambda names: (names[1] + "|" + names[0]) if names[1] else names[0]
    )

    if sport == "nhl" and style == DFSContestStyle.CLASSIC:
        three_player_line_goal_pctl = _get_nhl_3player_line_goal_pct(session, slate_to_games)
        player_scores_df = player_scores_df.join(three_player_line_goal_pctl)
    elif sport == "nba":
        lchu = _get_nba_low_cost_high_use(session, slate_to_games)
        player_scores_df = player_scores_df.join(lchu)

    return player_scores_df


def _create_inference_df(
    style: DFSContestStyle,
    teams_contest_df: pd.DataFrame,
    team_score_df: pd.DataFrame,
    player_scores_df: pd.DataFrame,
    slate_scores: dict[int, SlateScoreItem],
) -> pd.DataFrame:
    """
    join contest, slate id, team score, player position scores, slate_scores
    """
    assert (
        len(teams_contest_df) >= len(team_score_df) == len(player_scores_df) >= len(slate_scores)
    ), (
        "expecting more team_contest rows (one for every known dfs contest entry) "
        "than team/player score data (one for every slate?) which should be more "
        "than the found slates (slate scoring will fail on opening day due to no season to date data)"
    )

    slate_scores_df = pd.DataFrame(
        list(slate_scores.values()),
        index=list(slate_scores.keys()),
        columns=SlateScoreItem._fields,
    )
    slate_scores_df = slate_scores_df.assign(
        top_possible_minus_rational=(
            slate_scores_df.top_possible_lineup_score - slate_scores_df.top_rational_lineup_score
        )
    )
    if slate_scores_df.addl_scoring.notna().any():
        addl_df = pd.DataFrame(
            [ss.addl_scoring for ss in slate_scores.values()], index=list(slate_scores.keys())
        )
        slate_scores_df = pd.concat([slate_scores_df, addl_df], axis=1)
    slate_scores_df = slate_scores_df.drop(
        columns=["addl_scoring", "rational_lineup_settings_index", "games_count"]
    )
    contest_cols = ["date", "style", "type", "top_score", "last_winning_score", "link", "slate_id"]
    if style == DFSContestStyle.SHOWDOWN:
        teams_contest_df = teams_contest_df.rename(columns={"entries": "contest_entries"})
        contest_cols.append("contest_entries")
    elif style == DFSContestStyle.CLASSIC:
        contest_cols.append("team_count")
    df = (
        teams_contest_df[contest_cols]
        .rename(columns={"top_score": "top_winning_score"})
        .join(slate_scores_df, on="slate_id")
        .join(team_score_df, on="slate_id")
        .join(player_scores_df, on="slate_id")
    )
    return df


def _generate_dataset(
    cfg,
    sport,
    service_name,
    style: DFSContestStyle,
    contest_type: Contest,
    contest_data_path,
    top_percentile,
    min_date: date,
    max_date: date,
    max_count: None | int = None,
    slate_score_cache_mode: SlateScoreCacheMode = "default",
    datapath: str = "data",
    screen_lineup_constraints_mode="fail",
) -> pd.DataFrame:
    """
    max_count: maximum number of slates to process
    min_date: inclusive
    max_date: not inclusive
    datapath: target directory for data
    contest_data_path: input data path
    top_score_cache_mode:
        'default'=load and use the cache,
        'overwrite'=overwrite all existing cache data if any exists
        'missing'=use all existing valid cache data, any cached failures will be rerun
    """
    # get dfs contests from scraped dataset
    contest_df = _get_contest_df(
        service_name, sport, style, contest_type, min_date, max_date, contest_data_path
    )
    if len(contest_df) == 0:
        raise DataNotAvailableException(f"No contest data found for {min_date=} {max_date=}")

    if contest_df is not None and max_count is not None:
        contest_df = contest_df.head(max_count)

    draft_df = _get_draft_df(service_name, sport, style, min_date, max_date, contest_data_path)

    teams_contest_df = _create_teams_contest_df(contest_df, draft_df, service_name, sport)
    assert len(teams_contest_df) > 0

    df_stat_names = get_stat_names(sport, _SERVICE_NAME_TO_ABBR[service_name], as_str=True)

    db_obj = db.get_db_obj(cfg["db_filename"], readonly=True)
    with db_obj.session_scoped() as session:
        slate_db_df = _get_slate_df(session, service_name, style, min_date, max_date)

        # note that some slates will not be found due to mismatches in games found in
        # cost files and the games in the the downloaded contest results, and other slates
        # may appear multiple times if multiple contests were entered (hence the drop_duplicates call
        team_count_and_slate_id_df = teams_contest_df.apply(
            _get_slate_id, axis=1, args=(slate_db_df,)
        ).assign(date=teams_contest_df["date"])
        teams_contest_new_cols = {"slate_id": team_count_and_slate_id_df.slate_id}
        if "team_count" in team_count_and_slate_id_df:
            teams_contest_new_cols["team_count"] = team_count_and_slate_id_df.team_count
        teams_contest_df = teams_contest_df.assign(**teams_contest_new_cols)

        if team_count_and_slate_id_df.slate_id.isna().any():
            defined_slates_df = team_count_and_slate_id_df.dropna(subset="slate_id")
            _LOGGER.warning(
                "Slate scoring features for %d of %s slates will be skipped because slate could not be matched in DB",
                len(teams_contest_df) - len(defined_slates_df),
                len(teams_contest_df),
            )
            if len(teams_contest_df) - len(defined_slates_df) == 0:
                raise DataNotAvailableException("No slates ids found (based on teams contest df)")
        else:
            defined_slates_df = team_count_and_slate_id_df

        # df with index of slate_id and columns team_count and date
        defined_slates_df = defined_slates_df.set_index("slate_id").drop_duplicates()

        if "team_count" in defined_slates_df:
            # sort slates by date (ascending) and slate side (descending) so that bigger
            # slates are done first on each date to help with caching
            slate_ids = defined_slates_df.sort_values(
                ["date", "team_count"], ascending=[True, False]
            ).index.astype(int)
        else:
            slate_ids = defined_slates_df.index.astype(int)

        slate_to_games = {
            slate.id: slate.games
            for slate in session.query(db.DailyFantasySlate).where(
                db.DailyFantasySlate.id.in_(slate_ids)
            )
        }

        team_score_df = _create_team_score_df(session, slate_to_games, top_percentile, style, sport)

        player_scores_df = _get_player_scores(
            session, cfg, sport, style, slate_to_games, top_percentile, df_stat_names
        )

        # cache for top scores
        with score_cache_ctx(
            sport, style, slate_score_cache_mode, cache_dir=datapath
        ) as slate_score_cache:
            # bpl=best possible lineup ; slate_id -> (bpl-true-score, bpl-true-score - bpl-pred-score, lchv_count)
            slate_scores = {}
            for slate_id in (tqdm_iter := tqdm(slate_ids, desc="slates")):
                slate_date_str = (
                    defined_slates_df[defined_slates_df.index == slate_id]
                    .iloc[0]["date"]
                    .date()
                    .strftime("%Y%m%d")
                )
                tqdm_iter.set_postfix_str(slate_date_str)
                ss = slate_scoring(
                    session,
                    slate_id,
                    score_cache=slate_score_cache,
                    screen_lineup_constraints_mode=screen_lineup_constraints_mode,
                )
                if ss:
                    slate_scores[int(slate_id)] = ss

    inf_df = _create_inference_df(
        style, teams_contest_df, team_score_df, player_scores_df, slate_scores
    )

    # test that inf_df has all features and fail if it does not
    inf_df = test_for_expected_cols(inf_df, sport, style, unexpected_mode="fail", features="all")

    filepath = os.path.join(
        datapath, f"{sport}-{service_name}-{style.name}-{contest_type.TYPE_NAME}.csv"
    )
    _LOGGER.info("Writing data to '%s'", filepath)
    inf_df.to_csv(filepath, index=False)
    return inf_df


def _get_date_range(cfg: dict, service: str) -> tuple[date | None, date | None]:
    min_date_by_service = cast(dict, cfg["min_date"])
    max_date_by_service = cast(dict, cfg["max_date"])
    min_date = (
        min_date_by_service.get(service, min_date_by_service.get(None))
        if isinstance(min_date_by_service, dict)
        else min_date_by_service
    )
    max_date = (
        max_date_by_service.get(service, max_date_by_service.get(None))
        if isinstance(max_date_by_service, dict)
        else max_date_by_service
    )
    return min_date, max_date


def xform(
    sport,
    cfg: dict[str, Any],
    services: list[str],
    styles: list[DFSContestStyle],
    contest_types,
    top_score_cache_mode,
    data_path,
    contest_data_path,
    top_percentile,
    date_override: tuple[date, date] | None,
):
    """
    returns a dict mapping tuples describing the data generated to dataframes with the data
    """
    dfs: dict[tuple, pd.DataFrame] = {}
    if date_override:
        min_date, max_date = date_override
    for service in (service_iter := tqdm(services, desc="dfs-service", disable=len(services) == 1)):
        service_iter.set_postfix_str(service)
        if date_override is None:
            min_date, max_date = _get_date_range(cfg, service)
        assert (min_date is None) or (max_date is None) or min_date <= max_date, (
            "invalid date range. max_date must be greater than min_date. Or one must be None"
        )
        assert DFSContestStyle.CLASSIC not in styles or styles[0] == DFSContestStyle.CLASSIC, (
            "if classic is in styles it should be first so that its lineup generation cache can be reused"
        )

        if min_date == max_date:
            max_date += timedelta(days=1)
        for style in (style_iter := tqdm(styles, desc="contest-style", disable=len(styles) == 1)):
            style_iter.set_postfix_str(style.name)
            for contest_type in (
                contest_type_iter := tqdm(
                    contest_types, desc="contest-type", disable=len(contest_types) == 1
                )
            ):
                contest_type_iter.set_postfix_str(contest_type.TYPE_NAME)
                _LOGGER.info(
                    "Processing sport=%s service=%s style=%s contest-type=%s",
                    sport,
                    service,
                    style,
                    contest_type.TYPE_NAME,
                )
                slcm = (
                    "warn" if (sport, service) in _WARN_ON_SCREEN_LINEUP_CONSTRAINTS_ERR else "fail"
                )
                try:
                    dfs[(sport, service, style, contest_type.TYPE_NAME)] = _generate_dataset(
                        cfg,
                        sport,
                        service,
                        style,
                        contest_type,
                        contest_data_path,
                        top_percentile,
                        min_date=min_date,
                        max_date=max_date,
                        slate_score_cache_mode=top_score_cache_mode,
                        screen_lineup_constraints_mode=slcm,
                        datapath=data_path,
                    )
                    _LOGGER.info(
                        "Finished processing sport=%s service=%s style=%s contest-type=%s. "
                        "%i rows in dataset",
                        sport,
                        service,
                        style,
                        contest_type.TYPE_NAME,
                        len(dfs[(sport, service, style, contest_type.TYPE_NAME)]),
                    )
                except DataNotAvailableException as ex:
                    _LOGGER.error(
                        "Error processing sport=%s service=%s style=%s contest-type=%s",
                        sport,
                        service,
                        style,
                        contest_type.TYPE_NAME,
                        exc_info=ex,
                    )

    _LOGGER.info("Finished with sport %s", sport)
    return dfs
