#!/usr/bin/env python
"""CLI tool for calculating venue features like park factor and true home field advantage.
See VenueFeatures.md for details"""

import argparse
import os
from typing import Literal

import pandas as pd
from fantasy_py import db, log
from fantasy_py.calculation import elo
from fantasy_py.sport.extra_stats import expected_elo_mov
from sqlalchemy import func, select
from tabulate import tabulate
from tqdm import tqdm

_WEIGHTS = [0.5, 0.33, 0.17]
"""weights used for weighted average from most recent season to least recent"""

_Features = Literal["pf", "thfa"]
"""the features that can be processed. pf=park-factor thfa=true-home-field-advantage"""

_FEATHURE_ABBR_TO_NAME: dict[_Features, str] = {"pf": "park-factor", "thfa": "true-home-field-ad"}
"""mapping of feature to string to use as part of filename"""


def _create_cli_parser():
    parser = argparse.ArgumentParser(description="Predictions for past/future games")
    parser.add_argument("DB_FILE", help="The database file")
    parser.add_argument(
        "feature",
        choices=_Features.__args__,
        help="What feature to generate data for. pf=park-factor thfa=true-home-field-adventage. "
        "See VenueFeatures.md for details",
    )

    parser.add_argument("--min_season", type=int)
    parser.add_argument("--max_season", type=int)

    output_group = parser.add_argument_group(
        "Output Group",
        "One of the following can be used to specify what to do with the output. "
        "If nothing is used then result will be printed to stdio",
    )
    me_output_group = output_group.add_mutually_exclusive_group()
    me_output_group.add_argument(
        "--parquet",
        nargs="?",
        const="",
        help="write the result to a parquet file at this path, "
        "if filepath is not given then default path+name will be used. "
        "default={$FANTASY_ARCHIVE_BASE}/{sport}/{feature}.pq",
        metavar="filepath",
    )
    me_output_group.add_argument(
        "--csv",
        nargs="?",
        const="",
        help="write the result to a csv file at this path"
        "if filepath is not given then default path+name will be used. "
        "default={$FANTASY_ARCHIVE_BASE}/{sport}/{feature}.csv",
        metavar="filepath",
    )

    return parser


def _get_scoring_data(db_obj, min_season: None | int, max_season: None | int):
    """
    return dataframe containing game results for all games played in the requested
    seasons.
    """
    _LOGGER.info("Loading scoring data min_season=%s max_season=%s", min_season, max_season)
    with db_obj.session_scoped() as session:
        stmt = select(
            db.Game.venue,
            db.Game.season,
            db.Game.game_number,
            db.Game.home_team_id,
            db.Game.away_team_id,
            db.Game.score_home,
            db.Game.score_away,
            db.Team.abbr.label("home_team_abbr"),
            db.Team.name.label("home_team_name"),
        ).where(db.Game.home_team_id == db.Team.id)
        if min_season:
            stmt = stmt.filter(db.Game.season >= min_season)
        if max_season:
            stmt = stmt.filter(db.Game.season <= max_season)
        df = pd.read_sql(stmt, session.bind)
    return df


def _get_elo_data(db_obj, min_season: None | int, max_season: None | int):
    """
    return home field advantage adjusted mov elo scores for the requested seasons

    returns dataframe with columns season, game_number, team_id, elo. sorted by season, game_number
    """
    _LOGGER.info("Loading elo data min_season=%s max_season=%s", min_season, max_season)
    home_team_adjustment = elo.HOME_FIELD_ADVANTAGE_MODIFIERS[db_obj.db_manager.ABBR]

    with db_obj.session_scoped() as session:
        stmt = (
            select(
                db.Game.season,
                db.Game.game_number,
                (db.Game.home_team_id == db.CalculationDatum.team_id).label("is_home"),
                db.CalculationDatum.team_id,
                db.CalculationDatum.value.label("elo"),
            )
            .join(db.Statistic, db.Statistic.id == db.CalculationDatum.statistic_id)
            .filter(db.Statistic.name == "elo", db.Game.id == db.CalculationDatum.game_id)
        )
        if min_season:
            stmt = stmt.filter(db.Game.season >= min_season)
        if max_season:
            stmt = stmt.filter(db.Game.season <= max_season)
        stmt = stmt.order_by(db.Game.season, db.Game.game_number)
        df = pd.read_sql(stmt, session.bind)

    df["elo"] = df.apply(lambda row: row.elo + (home_team_adjustment if row.is_home else 0), axis=1)
    return df


def _true_home_field_advantage(db_obj, min_season: int, max_season: int):
    """
    return a dataframe with true home field advantage data for the requested seasons. dataframe columns
    will include [season, venue, score, home_team_id, home_team_name, home_team_abbr] where, for each row,
    score is the estimated true home field advantage that should be used for that team when playing at home
    where home is the venue on that row and the home team is the same as described on that row.
    """
    _LOGGER.info(
        "Calculating venue true home field advantage for min-season=%s max_season=%s",
        min_season,
        max_season,
    )
    df = _get_scoring_data(db_obj, min_season, max_season)
    _LOGGER.info(
        "%i games to process over seasons %i - %i", len(df), df.season.min(), df.season.max()
    )
    elo_df = _get_elo_data(db_obj, min_season, max_season)

    def _get_elo(row, ha: Literal["home", "away"]):
        """for the game in the row return the starting elo for the requested team"""
        if row.game_number == 1:
            return 1500
        team_id = row[ha + "_team_id"]
        elo_row = elo_df.query(
            "season == @row.season and team_id == @team_id and game_number < @row.game_number"
        )
        if len(elo_row) == 0:
            return 1500
        return elo_row.iloc[-1].elo

    # 1. Calculate Game-Level Venue Performance
    # Fetch Elo for both teams going into the game
    tqdm.pandas(desc="get-home-elo-pregame")
    home_elo = df.progress_apply(_get_elo, args=("home",), axis=1)
    tqdm.pandas(desc="get-away-elo-pregame")
    away_elo = df.progress_apply(_get_elo, args=("away",), axis=1)
    df_w_elo = df.assign(home_elo=home_elo, away_elo=away_elo)

    # Expected margin (Neutral site assumption)
    _LOGGER.info("calculating expected home margin-of-victory for all games")
    expected_home_mov = df_w_elo.apply(
        (lambda row: expected_elo_mov(db_obj.db_manager.ABBR, row.home_elo, row.away_elo)), axis=1
    )
    actual_margin = df_w_elo.score_home - df_w_elo.score_away
    df_w_elo_and_mov = df_w_elo.assign(
        expected_home_mov=expected_home_mov, actual_home_mov=actual_margin
    )

    # Isolated venue effect for this specific game
    historical_perf = df_w_elo_and_mov.assign(
        venue_effect=df_w_elo_and_mov.actual_home_mov - df_w_elo_and_mov.expected_home_mov
    )

    # 2. Aggregate and Add Padding for Inference season
    venue_stats = (
        historical_perf.groupby(
            ["venue", "season", "home_team_id", "home_team_abbr", "home_team_name"]
        )
        .venue_effect.mean()
        .reset_index()
    )

    final_season = df.season.max()
    final_season_home_teams = df.query("season == @final_season")[
        ["venue", "home_team_id", "home_team_abbr", "home_team_name"]
    ].drop_duplicates()
    pad_df = final_season_home_teams.assign(season=final_season + 1)

    # Combine and Sort
    full_venue_df = pd.concat([venue_stats, pad_df], ignore_index=True)
    full_venue_df = full_venue_df.sort_values(["venue", "season"])

    # 3. Apply  Weights
    # Create shifted columns (Lags) for each venue
    for i in range(1, len(_WEIGHTS) + 1):
        full_venue_df[f"v_lag{i}"] = full_venue_df.groupby(
            ["venue", "home_team_id"]
        ).venue_effect.shift(i)

    full_venue_df["score"] = sum(
        full_venue_df[f"v_lag{i}"].fillna(0) * weight for i, weight in enumerate(_WEIGHTS, 1)
    )

    return full_venue_df[
        ["venue", "season", "score", "home_team_id", "home_team_abbr", "home_team_name"]
    ]


def _park_factor(db_obj, min_season: None | int, max_season: None | int):
    df = _get_scoring_data(db_obj, min_season, max_season)

    # 1. Calculate Raw Annual Park Factor
    # Total runs per game at each venue (Home + Opponent)
    home_stats = (
        df.groupby(["venue", "season", "home_team_id"])
        .apply(lambda x: (x["score_home"] + x["score_away"]).mean())
        .reset_index(name="avg_runs_home")
    )

    # Total runs per game for each team when they are away
    away_stats = (
        df.groupby(["season", "away_team_id"])
        .apply(lambda x: (x["score_home"] + x["score_away"]).mean())
        .reset_index(name="avg_runs_away")
    )

    # Merge to get the 1-year Raw PF
    raw_pf = pd.merge(
        home_stats,
        away_stats,
        left_on=["season", "home_team_id"],
        right_on=["season", "away_team_id"],
    )
    raw_pf["raw_pf"] = raw_pf["avg_runs_home"] / raw_pf["avg_runs_away"]

    # handle rare case of teams sharing a venue
    raw_pf = raw_pf.groupby(["venue", "season"])["raw_pf"].mean().reset_index()

    # 2. Setup N-Year Rolling Weighted Average
    # Add padding rows for inference season and Sort to ensure LAG/Shift operations align correctly
    final_season = df.season.max()
    final_season_venues = df.query("season == @final_season").venue.unique()
    pad_df = pd.DataFrame(
        {"season": [final_season + 1] * len(final_season_venues), "venue": final_season_venues}
    )
    padded_df = pd.concat([raw_pf, pad_df], ignore_index=False).sort_values(["venue", "season"])

    # Create shifted columns (Lags) for each venue
    for i in range(1, len(_WEIGHTS) + 1):
        padded_df[f"pf_lag{i}"] = padded_df.groupby("venue")["raw_pf"].shift(i)

    # 3. Apply Weights
    padded_df["score"] = sum(
        padded_df[f"pf_lag{i}"].fillna(1.0) * weight for i, weight in enumerate(_WEIGHTS, 1)
    )

    # Final Result
    result_df = padded_df[["venue", "season", "score"]].dropna()
    return result_df


def _initialize(parser, cli_args):
    """
    process the command line args and return (db_obj, output-filepath, min_season, max_season)
    """
    _LOGGER.info("Loading '%s'", cli_args.DB_FILE)
    db_obj = db.get_db_obj(cli_args.DB_FILE)

    with db_obj.session_scoped() as session:
        (db_min_season, db_max_season) = session.execute(
            select(func.min(db.Game.season), func.max(db.Game.season))
        ).one()

    if cli_args.min_season:
        if not (db_min_season <= cli_args.min_season <= db_max_season):
            parser.error(
                f"{cli_args.min_season=} is not between db min and max seasons {db_min_season=} {db_max_season=}"
            )
        min_season = cli_args.min_season
    else:
        min_season = db_min_season

    if cli_args.max_season:
        if not (db_min_season <= cli_args.max_season <= db_max_season):
            parser.error(
                f"{cli_args.max_season=} is not between db min and max seasons {db_min_season=} {db_max_season=}"
            )
        max_season = cli_args.max_season
    else:
        max_season = db_max_season

    if cli_args.csv is None and cli_args.parquet is None:
        _LOGGER.info("Result will be written to stdout")
        return db_obj, None, min_season, max_season

    output_filepath = cli_args.csv if isinstance(cli_args.csv, str) else cli_args.parquet
    if len(output_filepath) == 0:
        # infer a filepath
        filetype = "csv" if cli_args.csv is not None else "pq"
        final_output_season = db_obj.db_manager.season_increment(max_season)
        filepath_parts = [
            os.environ["FANTASY_ARCHIVE_BASE"],
            db_obj.db_manager.ABBR,
            f"{_FEATHURE_ABBR_TO_NAME[cli_args.feature]}.{min_season}-{final_output_season}.{filetype}",
        ]
        output_filepath = os.path.join(*filepath_parts)

    _LOGGER.info("Result will be written to '%s'", output_filepath)
    return db_obj, output_filepath, min_season, max_season


if __name__ == "__main__":
    log.set_default_log_level(only_fantasy=False)
    _LOGGER = log.get_logger(__name__)
    cli_parser = _create_cli_parser()
    cli_args = cli_parser.parse_args()

    _LOGGER.info("*** %s processing ***", _FEATHURE_ABBR_TO_NAME[cli_args.feature])

    db_obj, output_filepath, min_season, max_season = _initialize(cli_parser, cli_args)

    if cli_args.feature == "pf":
        df = _park_factor(db_obj, cli_args.min_season, cli_args.max_season)
    elif cli_args.feature == "thfa":
        df = _true_home_field_advantage(db_obj, cli_args.min_season, cli_args.max_season)
    else:
        cli_parser.error(f"feature={cli_args.feature} not implemented!")

    if output_filepath is None:
        print(tabulate(df, headers="keys", showindex=False))
    elif cli_args.parquet is not None:
        _LOGGER.info("Writing result to parquet file at '%s'", output_filepath)
        df.to_parquet(output_filepath)
    elif cli_args.csv is not None:
        if df.venue.str.contains(",").any():
            raise ValueError("Commas found in some venue names. CSV dump failed.")
        _LOGGER.info("Writing result to csv file at '%s'", output_filepath)
        df.to_csv(output_filepath, index=False)
    else:
        cli_parser.error(
            f"Not sure how we got here. Request was to output to file '{output_filepath}' but "
            "no output format was identified"
        )

    _LOGGER.success("*** Finished %s Processing ***", _FEATHURE_ABBR_TO_NAME[cli_args.feature])
