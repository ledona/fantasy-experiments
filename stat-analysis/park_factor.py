#!/usr/bin/env python
"""CLI tool for calculating park factor. See ParkFactor.md for details"""

import argparse

import pandas as pd
from fantasy_py import db, log
from sqlalchemy import select
from tabulate import tabulate

_WEIGHTS = [0.55, 0.33, 0.17]
"""weights used for weighted average from most recent season to least recent"""


def _create_cli_parser():
    parser = argparse.ArgumentParser(description="Predictions for past/future games")
    parser.add_argument("DB_FILE", help="The database file")

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


def _main(args):
    log.set_default_log_level(only_fantasy=False)
    logger = log.get_logger(__name__)

    logger.info("Loading '%s'", args.DB_FILE)
    db_obj = db.get_db_obj(args.DB_FILE)

    logger.info(
        "Loading scoring data min_season=%s max_season=%s", args.min_season, args.max_season
    )
    with db_obj.session_scoped() as session:
        stmt = select(
            db.Game.venue,
            db.Game.season,
            db.Game.home_team_id,
            db.Game.away_team_id,
            db.Game.score_home,
            db.Game.score_away,
        )
        if args.min_season:
            stmt = stmt.filter(db.Game.season >= args.min_season)
        if args.max_season:
            stmt = stmt.filter(db.Game.season <= args.max_season)
        df = pd.read_sql(stmt, session.bind)

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
    padded_df["rolling_park_factor"] = sum(
        padded_df[f"pf_lag{i}"] * weight for i, weight in enumerate(_WEIGHTS, 1)
    )

    # Final Result
    result_df = padded_df[["venue", "season", "rolling_park_factor"]].dropna()

    if args.parquet:
        logger.info("Writing result to parquet file at '%s'", args.parquet)
        result_df.to_parquet(args.parquet)
    elif args.csv:
        if result_df.venue.str.contains(",").any():
            raise ValueError("Commas found in some venue names. CSV dump failed.")
        logger.info("Writing result to csv file at '%s'", args.csv)
        result_df.to_csv(args.csv, index=False)
    else:
        print(tabulate(result_df, showindex=False))


if __name__ == "__main__":
    cli_parser = _create_cli_parser()
    cli_args = cli_parser.parse_args()
    _main(cli_args)
