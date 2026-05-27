import logging
import os
import re
from datetime import date
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

from .slate_scoring import (
    SlateScoreCacheMode,
    get_stat_names,
    score_cache_ctx,
    slate_scoring,
    SlateScoreItem,
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


def _create_team_score_df(session: Session, slate_ids_str, percentile, style: DFSContestStyle):
    """return a dataframe with final game score (e.g. mlb run total) features"""
    sql = f"""
    select distinct daily_fantasy_slate.id as slate_id, game.id as game_id
           ,game.score_home, game.score_away
           ,(game.score_home + game.score_away) as total_score
           ,max(game.score_home, game.score_away) as high_score
           ,abs(game.score_home - game.score_away) as margin_of_victory
    from daily_fantasy_slate
        join daily_fantasy_cost on daily_fantasy_slate.id = daily_fantasy_cost.daily_fantasy_slate_id
        join game on ((game.date = daily_fantasy_slate.date or 
		               game.dt between daily_fantasy_slate.date and datetime(daily_fantasy_slate.date, '+1 days', '+6 hours')) and
                      game.season = daily_fantasy_slate.season and 
                      (daily_fantasy_cost.team_id in (game.away_team_id, game.home_team_id)))
    where daily_fantasy_slate.id in ({slate_ids_str})
    """
    db_team_score_df = pd.read_sql_query(sql, session.connection())
    if len(db_team_score_df) == 0:
        raise DataNotAvailableException(
            f"Empty team score df when retrieving for slates: {slate_ids_str}"
        )

    if style.name == "CLASSIC":
        melted_db_df = db_team_score_df.melt(
            id_vars=["slate_id", "game_id"], value_vars=["score_home", "score_away"]
        )
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
        if session.info["fantasy.db_manager"].ABBR == "mlb":
            high_score = db_team_score_df.groupby("slate_id")["high_score"].apply(max)
            team_score_df = team_score_df.join(high_score)

        return team_score_df

    if style.name == "SHOWDOWN":
        if db_team_score_df.game_id.duplicated().any():
            raise UnexpectedValueError("Showdown slate data should not have any duplicated games")
        cols = ["slate_id", "total_score", "margin_of_victory"]
        if session.info["fantasy.db_manager"].ABBR == "mlb":
            cols.append("high_score")
        team_score_df = db_team_score_df[cols]
        return team_score_df.set_index("slate_id")

    raise NotImplementedError()


def _get_exploded_pos_df(
    session,
    sport,
    slate_ids_str,
    cost_pos_drop: None | set,
    cost_pos_rename: None | dict,
    df_stat_names,
):
    # For mlb double headers this query will cause inaccuracy for players
    # that played in both games have a date equal to the slate date or
    # must have a datetime starting prior to 6am on the following date
    # To fix, game_id will need to be accurately/specifically linked to
    # slate (as opposed to assuming that the only game on a date for a
    # player/team is the same game that is in the slate.
    sql = f"""
    select daily_fantasy_slate.id as slate_id, positions as cost_positions, 
        player_position.abbr as stat_position, 
        value as score, daily_fantasy_cost.team_id, daily_fantasy_cost.player_id
    from daily_fantasy_slate
        join daily_fantasy_cost on 
           daily_fantasy_slate.id = daily_fantasy_cost.daily_fantasy_slate_id
        join game on (
           (game.date = daily_fantasy_slate.date or 
		    game.dt between daily_fantasy_slate.date and datetime(daily_fantasy_slate.date, '+1 days', '+6 hours')) and
           game.season = daily_fantasy_slate.season and 
           (daily_fantasy_cost.team_id in (game.away_team_id, game.home_team_id))
        )
        join calculation_datum on (
            calculation_datum.game_id = game.id and 
            calculation_datum.player_id is daily_fantasy_cost.player_id and
            calculation_datum.team_id = daily_fantasy_cost.team_id
        )
        join statistic on calculation_datum.statistic_id = statistic.id
        join player on daily_fantasy_cost.player_id = player.id
        join player_position on player.player_position_id = player_position.id
    where daily_fantasy_slate.id in ({slate_ids_str}) and
        statistic.name in ({df_stat_names})
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
    if missing_cols := _EXPECTED_CONTEST_COLS.difference(contest_df.keys()):
        raise DataNotAvailableException(
            f"Not all expected contest data columns were found. Missing cols={missing_cols}"
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


def _create_team_contest_df(contest_df, draft_df, service, sport):
    service_cls = CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, service)
    abbr_remaps = service_cls.TEAM_ABBR_ALIASES.get(sport)

    # add team/lineup draft data
    team_contest_df = pd.merge(contest_df, draft_df, on="contest_id")
    team_contest_df.team_abbr = team_contest_df.team_abbr.map(
        lambda abbr: (abbr_remaps.get(abbr) or abbr) if abbr_remaps else abbr
    )

    return team_contest_df


def _common_title(title_series: pd.Series) -> str:
    """the title of a contest will be the common prefix amongst all the possible contest titles"""
    title_list = title_series.tolist()
    return "" if None in title_list else os.path.commonprefix(title_list)


def _create_teams_contest_df(contest_df, draft_df, service, sport):
    """group contests together and create team sets used in each contest"""
    tc_df = _create_team_contest_df(contest_df, draft_df, service, sport)

    tc_df = pd.DataFrame(
        tc_df.groupby(["contest_id", "date", "style", "type", "link", "entries"]).agg(
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

    slate_db_df["team-count"] = slate_db_df.teams.map(len)
    return slate_db_df


_NO_SLATE_ID_FOUND = pd.Series({"slate_id": None, "team-count": None})


def _get_slate_id(contest_row, slate_db_df) -> pd.Series:
    """
    guesses the db slate id for the contest_row
    returns - series of (slate_id, number of teams playing in slate)
    """
    try:
        date_slates = slate_db_df.loc[[contest_row.date]].sort_values("team-count")
    except KeyError:
        _LOGGER.warning("get_slate_id::Key error/No slates found for %s", contest_row.date)
        return _NO_SLATE_ID_FOUND
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

    return slates.iloc[0][["slate_id", "team-count"]]


def _get_position_scores(session, cfg, sport, slate_ids_str, top_percentile, df_stat_names):
    """return a dataframe containing the top_percentileth player dfs score for
    each position for the requested slate(s), sport and service"""
    db_exploded_pos_df = _get_exploded_pos_df(
        session,
        sport,
        slate_ids_str,
        cfg.get("cost_pos_drop"),
        cfg.get("cost_pos_rename"),
        df_stat_names,
    )

    db_pos_scores_df = (
        db_exploded_pos_df[["slate_id", "position", "score"]]
        .groupby(["slate_id", "position"])
        .agg(["median", lambda x: np.percentile(x, top_percentile * 100)])
    )
    db_pos_scores_df.columns = ["med-dfs", f"{top_percentile * 100}th-pctl-dfs"]
    db_pos_scores_df = db_pos_scores_df.reset_index(level="position").pivot(
        columns="position",
        values=["med-dfs", f"{top_percentile * 100}th-pctl-dfs"],
    )
    return db_pos_scores_df


def _create_inference_df(
    sport,
    style: DFSContestStyle,
    teams_contest_df: pd.DataFrame,
    slate_ids_df: pd.DataFrame,
    team_score_df: pd.DataFrame,
    db_pos_scores_df: pd.DataFrame,
    slate_scores: dict[int, SlateScoreItem],
) -> pd.DataFrame:
    """
    join contest, slate id, team score, player position scores, slate_scores
    """
    assert (
        len(teams_contest_df)
        == len(slate_ids_df)
        == len(team_score_df)
        == len(db_pos_scores_df)
        == len(slate_scores)
    ), "all of these should be the same length"

    db_pos_scores_df.columns = db_pos_scores_df.columns.map(lambda names: names[1] + "|" + names[0])
    slate_scores_df = pd.DataFrame(
        list(slate_scores.values()),
        index=list(slate_scores.keys()),
        columns=SlateScoreItem._fields,
    )

    df = (
        teams_contest_df[["date", "style", "type", "top_score", "last_winning_score", "link"]].rename(columns={'top_score': 'top_winning_score'})
        .join(slate_ids_df)
        .join(slate_scores_df)
        .join(team_score_df)
        .join(db_pos_scores_df)
    )
    raise NotImplementedError("""
Add the following features
1) low cost player value density
2) mean diff between true and predicted scores for 'chalk' players (e.g. top 15 highest projected players)
3) optimal 'rational' score (i.e. highest score using typical strategy by use actual scores instead of projected)
4) sanity gap (diff between best possible score and optimal rational score)
5) top 3 game actual totals (or actual total for a showdown)
6) NBA - minimum salaries players with 25+ mins
7) MLB - max team score, teams with 8+ runs, optimal lineup pitcher score
8) NFL - overall touchdown/yardage, % points contributed by DST/K in optimal lineup
9) NHL - goals scored by same line
                              
for showdown add
1) score differential
2) total contest entries
3) position of optimal captain
4) winning team actual score
5) (NBA) raw actual DFS score for the highest cost player
6) (NHL) optimal goalie dfs score
7) (MLB) optimal pitcher dfs score
            """)
    return df


def _get_expected_features(sport, style: DFSContestStyle) -> set[str]:
    raise NotImplementedError()


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
    if contest_df is not None and max_count is not None:
        contest_df = contest_df.head(max_count)

    draft_df = _get_draft_df(service_name, sport, style, min_date, max_date, contest_data_path)

    teams_contest_df = _create_teams_contest_df(contest_df, draft_df, service_name, sport)
    assert len(teams_contest_df) > 0

    df_stat_names = get_stat_names(sport, _SERVICE_NAME_TO_ABBR[service_name], as_str=True)

    db_obj = db.get_db_obj(cfg["db_filename"], readonly=True)
    with db_obj.session_scoped() as session:
        slate_db_df = _get_slate_df(session, service_name, style, min_date, max_date)

        team_count_and_slate_id_df = teams_contest_df.apply(
            _get_slate_id, axis=1, args=(slate_db_df,)
        )
        teams_contest_df = teams_contest_df.set_index(team_count_and_slate_id_df.slate_id)
        if team_count_and_slate_id_df.slate_id.isna().any():
            raise UnexpectedValueError("dataframe should not have any NAs for slate id")
        if len(team_count_and_slate_id_df) == 0:
            raise DataNotAvailableException("No slates ids found (based on teams contest df)")

        team_count_and_slate_id_df = team_count_and_slate_id_df.set_index("slate_id")
        slate_ids = team_count_and_slate_id_df.index.astype(int)
        slate_ids_str = ",".join(map(str, slate_ids))

        team_score_df = _create_team_score_df(session, slate_ids_str, top_percentile, style)

        pos_scores_df = _get_position_scores(
            session, cfg, sport, slate_ids_str, top_percentile, df_stat_names
        )

        # cache for top scores
        with score_cache_ctx(sport, slate_score_cache_mode, cache_dir=datapath) as score_dict:
            # bpl=best possible lineup ; slate_id -> (bpl-true-score, bpl-true-score - bpl-pred-score, lchv_count)
            slate_to_scores_func = partial(
                slate_scoring,
                session,
                score_cache=score_dict,
                screen_lineup_constraints_mode=screen_lineup_constraints_mode,
            )
            slate_scores = {
                int(slate_id): slate_to_scores_func(slate_id)
                for slate_id in tqdm(slate_ids, desc="slates")
            }

    inf_df = _create_inference_df(
        sport,
        style,
        teams_contest_df,
        team_count_and_slate_id_df,
        team_score_df,
        pos_scores_df,
        slate_scores,
    )

    expected_features = _get_expected_features(sport, style)

    if missing_cols := expected_features.difference(inf_df.columns):
        raise UnexpectedValueError(f"inf_df is missing columns: {sorted(missing_cols)}")

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
    styles,
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
        assert (min_date is None) or (max_date is None) or min_date < max_date, (
            "invalid date range. max_date must be greater than min_date. Or one must be None"
        )

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
