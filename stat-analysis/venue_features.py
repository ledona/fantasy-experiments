#!/usr/bin/env python
"""CLI tool for calculating venue features like park factor and true home field advantage.
See VenueFeatures.md for details"""

import argparse
from typing import Literal

import pandas as pd
from fantasy_py import db, log
from sqlalchemy import select
from tabulate import tabulate

_WEIGHTS = [0.5, 0.33, 0.17]
"""weights used for weighted average from most recent season to least recent"""


def _create_cli_parser():
    parser = argparse.ArgumentParser(description="Predictions for past/future games")
    parser.add_argument("DB_FILE", help="The database file")
    parser.add_argument(
        "feature",
        choices=["pf", "thfa"],
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
        "--parquet", help="write the result to a parquet file at this path", metavar="filepath"
    )
    me_output_group.add_argument(
        "--csv", help="write the result to a csv file at this path", metavar="filepath"
    )

    return parser


def _get_scoring_data(db_obj, min_season: None | int, max_season: None | int):
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
        )
        if min_season:
            stmt = stmt.filter(db.Game.season >= min_season)
        if max_season:
            stmt = stmt.filter(db.Game.season <= max_season)
        df = pd.read_sql(stmt, session.bind)
    return df


def _get_elo_data(db_obj, min_season: None | int, max_season: None | int):
    """
    return elo scores for the requested seasons, dataframe with
    columns season, game_number, team_id, elo. sorted by season, game_number
    """
    _LOGGER.info("Loading elo data min_season=%s max_season=%s", min_season, max_season)
    with db_obj.session_scoped() as session:
        stmt = (
            select(
                db.Game.season,
                db.Game.game_number,
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
    return df


_ELO_POINTS_TO_SPORT = {"nfl": 25, "nhl": 60, "nba": 15}


def _true_home_field_advantage(db_obj, min_season: None | int, max_season: None | int):
    df = _get_scoring_data(db_obj, min_season, max_season)
    elo_df = _get_elo_data(db_obj, min_season, max_season)

    def _get_elo(row, ha: Literal["home", "away"]):
        team_id = row[ha + "_team_id"]
        elo_row = elo_df.query(
            "season == @row.season and team_id == @team_id and game_number < @row.game_number"
        ).iloc[-1]
        return elo_row.elo

    # 1. Calculate Game-Level Venue Performance
    # Fetch Elo for both teams going into the game
    home_elo = df.apply(_get_elo, args=("home",), axis=1)
    away_elo = df.apply(_get_elo, args=("away",), axis=1)
    df_w_elo = df.assign(home_elo=home_elo, away_elo=away_elo)

    # Expected margin (Neutral site assumption)
    expected_margin = (df_w_elo["home_elo"] - df_w_elo["away_elo"]) / _ELO_POINTS_TO_SPORT[
        db_obj.db_manager.ABBR
    ]
    actual_margin = df_w_elo["home_score"] - df_w_elo["away_score"]
    df_w_elo_and_margin = df_w_elo.assign(
        expected_margin=expected_margin, actual_margin=actual_margin
    )

    # Isolated venue effect for this specific game
    historical_perf = df_w_elo_and_margin.assign(
        venue_perf=df_w_elo_and_margin["actual_margin"] - df_w_elo_and_margin["expected_margin"]
    )

    # 2. Aggregate and Add Padding for Inference season
    venue_stats = historical_perf.groupby(["venue", "season"])["venue_perf"].mean().reset_index()

    final_season = df.season.max()
    final_season_venues = df.query("season == @final_season").venue.unique()
    pad_df = pd.DataFrame(
        {"season": [final_season + 1] * len(final_season_venues), "venue": final_season_venues}
    )

    # Combine and Sort
    full_venue_df = pd.concat([venue_stats, pad_df], ignore_index=True)
    full_venue_df = full_venue_df.sort_values(["venue", "season"])

    # 3. Apply  Weights
    # Create shifted columns (Lags) for each venue
    for i in range(1, len(_WEIGHTS) + 1):
        full_venue_df[f"v_lag{i}"] = full_venue_df.groupby("venue")["raw_score"].shift(i)

    full_venue_df["score"] = sum(
        full_venue_df[f"v_lag{i}"].fillna(0) * weight for i, weight in enumerate(_WEIGHTS, 1)
    )

    return full_venue_df[["venue", "season", "score"]]


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

    # 3. Apply  Weights
    padded_df["score"] = sum(
        padded_df[f"pf_lag{i}"] * weight for i, weight in enumerate(_WEIGHTS, 1)
    )

    # Final Result
    result_df = padded_df[["venue", "season", "score"]].dropna()
    return result_df


if __name__ == "__main__":
    log.set_default_log_level(only_fantasy=False)
    _LOGGER = log.get_logger(__name__)
    cli_parser = _create_cli_parser()
    cli_args = cli_parser.parse_args()

    _LOGGER.info("Loading '%s'", cli_args.DB_FILE)
    db_obj = db.get_db_obj(cli_args.DB_FILE)

    if cli_args.feature == "pf":
        df = _park_factor(db_obj, cli_args.min_season, cli_args.max_season)
    elif cli_args.feature == "thfa":
        df = _true_home_field_advantage(db_obj, cli_args.min_season, cli_args.max_season)
    else:
        cli_parser.error(f"feature={cli_args.feature} not implemented!")

    if cli_args.parquet:
        _LOGGER.info("Writing result to parquet file at '%s'", cli_args.parquet)
        df.to_parquet(cli_args.parquet)
    elif cli_args.csv:
        if df.venue.str.contains(",").any():
            raise ValueError("Commas found in some venue names. CSV dump failed.")
        _LOGGER.info("Writing result to csv file at '%s'", cli_args.csv)
        df.to_csv(cli_args.csv, index=False)
    else:
        print(tabulate(df, showindex=False))
