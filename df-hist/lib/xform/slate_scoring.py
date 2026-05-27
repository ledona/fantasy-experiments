import argparse
import json
import math
import os
from argparse import Namespace
from contextlib import contextmanager
from numbers import Number
from typing import Literal, cast, NamedTuple

import numpy as np
import pandas as pd
from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    DataNotAvailableException,
    UnexpectedValueError,
    db,
    log,
)
from fantasy_py.lineup import FantasyCostAggregate, FantasyService, LineupGenerationFailure
from fantasy_py.lineup.knapsack import MixedIntegerKnapsackSolver
from fantasy_py.sport import SportDBManager

_LOGGER = log.get_logger(__name__)


SlateScoreCacheMode = Literal["default", "overwrite", "missing"]


def _score_lineup(
    db_obj,
    fca,
    solver,
    service_cls,
    slate_name,
    slate_info,
    epoch,
    screen_lineup_constraints_mode,
):
    """
    calculate the best possible lineup score for a historic slate

    returns (best-possible-lineup, score diff between actual score and predicted score, hist and pred player scores)
    """
    model_names = (service_cls.DEFAULT_MODEL_NAMES.get(db_manager.ABBR),)

    params = GenLineupsParams(n_lineups=1, score_data_type="historic")
    lineups, score_data = gen_lineups(
        db_obj,
        fca,
        model_names,
        solver,
        service_cls,
        params,
        slate=slate_name,
        slate_info=slate_info,
        slate_epoch=epoch,
        screen_lineup_constraints_mode=screen_lineup_constraints_mode,
        scores_to_include=["predicted"],
    )

    top_predicted_players_scores = (
        pd.merge(
            score_data["historic"].rename(columns={"fpts": "hist-fpts"}),
            score_data["predicted"].rename(columns={"fpts": "pred-fpts"}),
            on=["game_id", "team_id", "player_id"],
            how="inner",
        )
        .sort_values("pred-fpts", ascending=False)
        .head(int(len(score_data["predicted"]) * _TOP_PREDICTED_PLAYER_PCTL / 100))
    )
    hist_pred_diff = float(
        top_predicted_players_scores["hist-fpts"].mean()
        - top_predicted_players_scores["pred-fpts"].mean()
    )
    return lineups[0], hist_pred_diff, score_data


def get_stat_names(sport, service_abbr: Literal["dk", "fd", "y"], as_str=False) -> str | list[str]:
    """
    returns stat names for the requested sport and service as either a comma seperated string that
    can be used in an sql query, or as a list of strings
    """
    stats: str | list[str]

    if sport == "nfl":
        stats = [f"{service_abbr}_score_off", f"{service_abbr}_score_def"]
    elif sport == "lol":
        stats = [f"{service_abbr}_match_score"]
    else:
        stats = [f"{service_abbr}_score"]

    if as_str:
        stats = "'" + "','".join(stats) + "'"
    return stats


_LOW_COST_HIGH_VALUE_COST_PCTL = 0.25
"""used to identify low cost players in a slate"""

_LOW_COST_HIGH_VALUE_SCORE_PCTL = 0.9
"""used to identify high scoring players"""


def _get_low_cost_high_value_count(
    slate_id: int, service, fca: FantasyCostAggregate, hist_scores_df: pd.DataFrame
) -> int:
    """return count of low cost players that overperformed in the slate"""

    def cost_func(player_id):
        player_dict = fca.get_mi_player(int(player_id))
        if not player_dict or "cost" not in player_dict:
            return None
        if isinstance(cost_val := player_dict["cost"], Number):
            return cost_val
        if service not in cost_val:
            return None
        if isinstance(service_cost := cost_val[service], Number):
            return service_cost
        if cost := service_cost.get(str(slate_id)):
            return cost

        raise NotImplementedError("don't know what else to try to get cost")

    cost_and_score_df = hist_scores_df.assign(cost=hist_scores_df.player_id.map(cost_func))

    # dataframe with cost and score thresholds for each slate
    min_score = np.percentile(cost_and_score_df.fpts, _LOW_COST_HIGH_VALUE_SCORE_PCTL * 100)
    max_cost = np.percentile(cost_and_score_df.cost, _LOW_COST_HIGH_VALUE_COST_PCTL * 100)

    low_cost_high_val_rows = cost_and_score_df[
        (cost_and_score_df.fpts > min_score) & (cost_and_score_df.cost < max_cost)
    ]

    return len(low_cost_high_val_rows)


class SlateScoreCacheItem(NamedTuple):
    """cache item for slate scoring"""

    top_lineup_score: float
    top_lineup_true_minus_pred: float
    """difference between true and predicted score for the top lineup"""
    top_rational_lineup_score: float
    low_cost_high_value_player_count: int


@contextmanager
def score_cache_ctx(sport: str, cache_mode: SlateScoreCacheMode, cache_dir="."):
    """context manager for caching lineup scoring results"""
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist")
    score_cache_filename = sport + "-slate.score.json"
    score_cache_filepath = os.path.join(cache_dir, score_cache_filename)

    score_dict: dict[int, SlateScoreCacheItem]

    orig_score_dict: dict[int, SlateScoreCacheItem] = {}

    if os.path.isfile(score_cache_filepath):
        if cache_mode in ("default", "missing"):
            with open(score_cache_filepath, "r") as f:
                cache_data = json.load(f)
            for slate_id, score in cache_data.items():
                if cache_mode == "missing" and score is None:
                    continue
                orig_score_dict[int(slate_id)] = SlateScoreCacheItem(*score)
        elif cache_mode == "overwrite":
            _LOGGER.info("Overwriting existing best score cache data at '%s'", score_cache_filepath)
        else:
            raise UnexpectedValueError("Unexpected lineup score cache mode", cache_mode)
    else:
        _LOGGER.info("Best score cache data not found! '%s'", score_cache_filepath)
        orig_score_dict = {}

    # make a copy so that we can figure out if there are updates
    # TODO: for diff, can probably do this more efficiently by comparing a hash of the before and after
    score_dict = dict(orig_score_dict)

    try:
        yield score_dict
    finally:
        if orig_score_dict != score_dict:
            # TODO: should save the cache as new scores are added
            _LOGGER.info("Writing updated best score values to cache '%s'", score_cache_filepath)
            with open(score_cache_filepath, "w") as f:
                json.dump(score_dict, f)
        _LOGGER.info("Exiting best_score_cache")


def _score_best_rational_lineup(db_obj, fca, screen_lineup_constraints_mode) -> float:
    """get scoring information for the best rational lineup"""
    raise NotImplementedError(
        """
        1) get the best possible score when sticking to typical lineup generation constraints (e.g. stacking)
        """
    )


def _score_top_lineup(
    db_obj,
    db_manager,
    game_date,
    fca: FantasyCostAggregate,
    starters,
    service_cls: FantasyService,
    slate_name,
    slate_id,
    constraints,
    screen_lineup_constraints_mode,
):
    """get scoring info for the best possible lineup"""
    solver = MixedIntegerKnapsackSolver(
        constraints.knapsack_constraints,
        constraints.budget,
        totals_func=constraints.totals_func,
        fill_all_positions=constraints.fill_all_positions,
    )

    epoch = db_manager.epoch_for_date(game_date)

    try:
        lineup, hist_pred_diff, scoring_data = _score_lineup(
            db_obj,
            fca,
            solver,
            service_cls,
            slate_name,
            starters.slates[slate_name],
            epoch,
            screen_lineup_constraints_mode,
        )
    except (DataNotAvailableException, LineupGenerationFailure) as ex:
        _LOGGER.warning(
            "Top Lineup generation failure for service_abbr='%s' sport='%s' slate '%s' (id=%i) on %s: %s",
            service_cls.ABBR,
            db_manager.ABBR,
            slate_name,
            slate_id,
            game_date,
            ex,
        )
        raise
    except Exception as ex:
        _LOGGER.error(
            "Unhandled error generating best lineup for service_abbr='%s' sport='%s' slate '%s' (id=%i) on %s.",
            service_cls.ABBR,
            db_manager.ABBR,
            slate_id,
            game_date,
            exc_info=ex,
        )
        raise

    hist_score = cast(float, lineup.historic_fpts)

    return hist_score, hist_pred_diff, scoring_data


def slate_scoring(
    session,
    slate_id,
    score_cache: None | dict[int, SlateScoreCacheItem] = None,
    screen_lineup_constraints_mode="fail",
):
    """
    Calculate the best possible fantasy score and difference between mean historic score
    vs predicted scores for top players for the requested slate.

    Function is used as a map function for a pandas series.

    pts_stats_names - the statistic names for the scores to use for players/teams
    best_score_cache - cache of slate ids mapped to their score. this will be
        searched and possibly updated to include the score for the requested slate

    returns - None if there is an error occurs, otherwise a tuple of
        (top-possible-lineup-score,
         top-possible-lineup-score - predicted-score-for-top-possible-lineup,
         top-rational-lineup-score,
         low-cost-high-value-player-count)
    """
    if not isinstance(slate_id, (int, float)) or math.isnan(slate_id):
        return [None] * 4

    slate_id = int(slate_id)
    if score_cache:
        if slate_id in score_cache:
            return score_cache[slate_id]
        _LOGGER.info("slate_id=%i not in best score cache", slate_id)

    db_manager = cast(SportDBManager, session.info["fantasy.db_manager"])

    # slate date
    slate = (
        session.query(db.DailyFantasySlate)
        .filter(db.DailyFantasySlate.id == int(slate_id))
        .one_or_none()
    )
    if slate is None:
        _LOGGER.warning("Error: Unable to find slate_id=%i in database", slate_id)
        return [None] * 4

    game_date = slate.date
    slate_name = slate.name
    service = slate.service

    _LOGGER.info(
        "Generating best historic lineup for %s slate '%s' (%i)", game_date, slate_name, slate_id
    )

    # get the starters
    starters_by_id = db_manager.get_starters(
        service, games_date=game_date, db_obj=session.info["db_obj"], slate=slate_id
    )
    if starters_by_id is None:
        raise DataNotAvailableException(f"Failed to retrieve starters for {game_date=} {slate_id=}")
    starters = starters_by_id.filter_by_slate(slate_name)
    if starters is None:
        raise UnexpectedValueError(
            f"{slate_name=} not in starters. Starters slates are {starters_by_id.slates.keys()}"
        )

    # TODO: most of the following should be defaults for the args object and should not be
    #   required here
    args = Namespace(
        starters_stale_mins=9999999,
        cache_dir=None,
        drop_games=None,
        no_fail=False,
        service=service,
        match_threshold=0.5,
        lineup_plan_paths=None,
        use_default_lineup_plans=True,
    )
    args, fca = db_manager.gen_lineups_preprocess(
        session.info["db_obj"], args, None, game_date, starters=starters, print_slate_info=False
    )[:2]

    service_cls = cast(FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, service))

    contest_constraints = service_cls.get_constraints(
        db_manager.ABBR, slate=starters.slates[slate_name]
    )
    assert contest_constraints is not None

    try:
        hist_score, hist_pred_diff, scoring_data = _score_top_lineup(
            session.info["db_obj"],
            db_manager,
            game_date,
            fca,
            starters,
            service_cls,
            slate_name,
            slate_id,
            contest_constraints,
            screen_lineup_constraints_mode,
        )
    except (DataNotAvailableException, LineupGenerationFailure) as ex:
        return None, None, None

    brl_score = _score_best_rational_lineup(
        session.info["db_obj"], fca, screen_lineup_constraints_mode
    )

    lchv_count = _get_low_cost_high_value_count(slate_id, service, fca, scoring_data["historic"])

    if score_cache:
        score_cache[slate_id] = SlateScoreCacheItem(
            hist_score, hist_pred_diff, brl_score, lchv_count
        )
    return hist_score, hist_pred_diff, brl_score, lchv_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the best lineup calculation")
    parser.add_argument("db_filename", help="database filename")
    parser.add_argument("service", help="service abbreviation")
    parser.add_argument("slate_id", help="slate id", type=int)

    _args = parser.parse_args()

    best_info = slate_scoring(_args.db_filename, _args.service, _args.slate_id)
    print(f"{best_info=}")
