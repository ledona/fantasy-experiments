import argparse
import json
import math
import os
from argparse import Namespace
from contextlib import contextmanager
from functools import cache
from numbers import Number
from typing import Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    DataNotAvailableException,
    JSONWithCommentsDecoder,
    UnexpectedValueError,
    db,
    log,
)
from fantasy_py.lineup import (
    FantasyCostAggregate,
    FantasyService,
    GenLineupsParams,
    LineupGenerationFailure,
    gen_lineups,
)
from fantasy_py.lineup.knapsack import MixedIntegerKnapsackSolver
from fantasy_py.sport import SportDBManager
from typeguard import check_type

_LOGGER = log.get_logger(__name__)


SlateScoreCacheMode = Literal["default", "overwrite", "missing"]


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

_TOP_PLAYERS_COUNT = {"classic": 15, "showdown": 2}
_TOP_PLAYERS_PCTL = 0.4


def _top_players_scoring_diff(score_data, contest_style) -> tuple[float, float]:
    """
    returns diff between actual and predicted scores for the top players where top
    players is defined as (top by percentile, and count)
    """
    top_players_sorted_by_pred = pd.merge(
        score_data["historic"].rename(columns={"fpts": "hist-fpts"}),
        score_data["predicted"].rename(columns={"fpts": "pred-fpts"}),
        on=["game_id", "team_id", "player_id"],
        how="inner",
    ).sort_values("pred-fpts", ascending=False)

    top_players_by_pctl = top_players_sorted_by_pred.head(
        int(len(score_data["predicted"]) * _TOP_PLAYERS_PCTL)
    )
    diff_for_pctl = float(
        top_players_by_pctl["hist-fpts"].mean() - top_players_by_pctl["pred-fpts"].mean()
    )

    top_n_players = top_players_sorted_by_pred.head(_TOP_PLAYERS_COUNT[contest_style])
    diff_for_n = float(top_n_players["hist-fpts"].mean() - top_n_players["pred-fpts"].mean())

    return diff_for_pctl, diff_for_n


def _slate_overperformances(slate_id: int, service, fca: FantasyCostAggregate, score_data):
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

    cost_and_score_df = score_data["historic"].assign(
        cost=score_data["historic"].player_id.map(cost_func)
    )

    # dataframe with cost and score thresholds for each slate
    min_score = np.percentile(cost_and_score_df.fpts, _LOW_COST_HIGH_VALUE_SCORE_PCTL * 100)
    max_cost = np.percentile(cost_and_score_df.cost, _LOW_COST_HIGH_VALUE_COST_PCTL * 100)

    low_cost_high_val_rows = cost_and_score_df[
        (cost_and_score_df.fpts > min_score) & (cost_and_score_df.cost < max_cost)
    ]

    return len(low_cost_high_val_rows)


class SlateScoreItem(NamedTuple):
    """scores for the slate"""

    top_possible_lineup_score: float
    """actual fantasy points scored by the best possible lineup"""
    top_rational_lineup_score: float
    """actual fantasy points scored by the lineup built using rational 
    lineup constuction strategies optimized with historic scoring"""
    low_cost_high_value_player_count: int
    """number of low cost players that significantly overperformed"""
    top_players_scoring_diff_n: float
    """mean diff between true and predicted scores of top n predicted players"""
    top_players_scoring_diff_pctl: float
    """mean diff between true and predicted scores of top percentile predicted players"""


@contextmanager
def score_cache_ctx(sport: str, cache_mode: SlateScoreCacheMode, cache_dir="."):
    """context manager for caching lineup scoring results"""
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist")
    score_cache_filename = sport + "-slate.score.json"
    score_cache_filepath = os.path.join(cache_dir, score_cache_filename)

    score_dict: dict[int, SlateScoreItem]

    orig_score_dict: dict[int, SlateScoreItem] = {}

    if os.path.isfile(score_cache_filepath):
        if cache_mode in ("default", "missing"):
            with open(score_cache_filepath, "r") as f:
                cache_data = json.load(f)
            for slate_id, score in cache_data.items():
                if cache_mode == "missing" and score is None:
                    continue
                orig_score_dict[int(slate_id)] = SlateScoreItem(*score)
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


def _score_lineup(
    db_obj,
    fca,
    solver,
    service_cls: type[FantasyService],
    slate_name,
    slate_id,
    slate_info,
    game_date,
    screen_lineup_constraints_mode,
    label: str,
    gen_lineup_params: GenLineupsParams,
):
    """
    calculate the best possible lineup score for a historic slate

    returns (best-possible-lineup, score diff between actual score and predicted score, hist and pred player scores)
    """
    model_names = service_cls.DEFAULT_MODEL_NAMES.get(db_obj.db_manager.ABBR)
    assert model_names
    epoch = db_obj.db_manager.epoch_for_date(game_date)

    try:
        lineups, score_data = gen_lineups(
            db_obj,
            fca,
            model_names,
            solver,
            service_cls,
            gen_lineup_params,
            slate=slate_name,
            slate_info=slate_info,
            slate_epoch=epoch,
            screen_lineup_constraints_mode=screen_lineup_constraints_mode,
            scores_to_include=["predicted"],
        )
    except (DataNotAvailableException, LineupGenerationFailure) as ex:
        _LOGGER.warning(
            "%s: Lineup generation failed for service_abbr='%s' sport='%s' slate '%s' (id=%i) on %s: %s",
            label,
            service_cls.ABBR,
            db_obj.db_manager.ABBR,
            slate_name,
            slate_id,
            game_date,
            ex,
        )
        raise
    except Exception as ex:
        _LOGGER.error(
            "%s: Unhandled error generating lineup for service_abbr='%s' sport='%s' slate '%s' (id=%i) on %s.",
            label,
            service_cls.ABBR,
            db_obj.db_manager.ABBR,
            slate_name,
            slate_id,
            game_date,
            exc_info=ex,
        )
        raise

    return float(lineups[0].historic_fpts), score_data


@cache
def _get_rational_lineup_gen_params(sport, contest_style):
    rational_params_filepath = os.path.join(
        os.path.dirname(__file__), "rational_lineup_params.json"
    )
    with open(rational_params_filepath, "r") as f_:
        all_rat_params = check_type(
            json.load(f_, cls=JSONWithCommentsDecoder), dict[str, list[dict]]
        )
    if rat_params_list := all_rat_params.get(f"{sport}-{contest_style}"):
        return rat_params_list
    raise ValueError(f"No rational lineup gen params defined for {sport=} {contest_style=}")


def slate_scoring(
    session,
    slate_id,
    score_cache: None | dict[int, SlateScoreItem] = None,
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
        raise UnexpectedValueError("id of slate is expected to be a number")
        # return None

    slate_id = int(slate_id)
    if score_cache:
        if slate_id in score_cache:
            return score_cache[slate_id]
        _LOGGER.info("slate_id=%i not in best score cache", slate_id)

    db_manager = cast(SportDBManager, session.info["fantasy.db_manager"])

    # slate date
    slate = cast(
        db.DailyFantasySlate,
        session.query(db.DailyFantasySlate)
        .filter(db.DailyFantasySlate.id == int(slate_id))
        .one_or_none(),
    )
    if slate is None:
        raise UnexpectedValueError("failed to find slate id in db")
        _LOGGER.warning("Error: Unable to find slate_id=%i in database", slate_id)
        return None

    game_date = slate.date
    slate_name = slate.name
    service = slate.service

    _LOGGER.info(
        "Generating best historic lineups for %s slate '%s' (%i)", game_date, slate_name, slate_id
    )

    # get the starters
    starters_by_id = db_manager.get_starters(
        service, games_date=game_date, db_obj=session.info["db_obj"], slate=slate_id
    )
    if starters_by_id is None:
        raise DataNotAvailableException(f"Failed to retrieve starters for {game_date=} {slate_id=}")
    starters = starters_by_id.filter_by_slate(slate_name)
    if starters is None or starters.slates is None:
        raise UnexpectedValueError(
            f"{slate_name=} not in starters. Starters slates "
            f"are {starters_by_id.slates.keys() if starters_by_id.slates else None}"
        )
    slate_info = starters.slates[slate_name]

    service_cls = cast(type[FantasyService], CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, service))
    args = Namespace(drop_games=None, no_fail=False, service=service, match_threshold=0.5)

    fca = db_manager.gen_lineups_preprocess(
        session.info["db_obj"],
        args,
        None,
        game_date,
        slate_name=slate_name,
        starters=starters,
        print_slate_info=False,
    )[1]
    contest_constraints = service_cls.get_constraints(db_manager.ABBR, slate=slate_info)
    assert contest_constraints is not None

    solver = MixedIntegerKnapsackSolver(
        contest_constraints.knapsack_constraints,
        contest_constraints.budget,
        totals_func=contest_constraints.totals_func,
        fill_all_positions=contest_constraints.fill_all_positions,
    )

    top_lineup_params = GenLineupsParams(score_data_type="historic", n_lineups=1)
    top_lineup_score, scoring_data = _score_lineup(
        session.info["db_obj"],
        fca,
        solver,
        service_cls,
        slate_name,
        slate_id,
        starters.slates[slate_name],
        game_date,
        screen_lineup_constraints_mode,
        "top-lineup",
        top_lineup_params,
    )

    contest_style = str(slate.style)
    for brl_gl_params_kwargs in _get_rational_lineup_gen_params(db_manager.ABBR, contest_style):
        brl_gl_params = GenLineupsParams(
            score_data_type="historic", n_lineups=1, **brl_gl_params_kwargs
        )
        brl_score = _score_lineup(
            session.info["db_obj"],
            fca,
            solver,
            service_cls,
            slate_name,
            slate_id,
            starters.slates[slate_name],
            game_date,
            screen_lineup_constraints_mode,
            "rational-lineup",
            brl_gl_params,
        )[0]
        break
    else:
        raise NotImplementedError(
            "Failed to generate a rational lineup after generating a top lineup "
            f"for sport={db_manager.ABBR} {game_date=} {slate_name=} {contest_style=}. "
            "Additional, less restrictive, rational lineup generation params are needed."
        )

    top_pctl_players_diff, top_n_players_diff = _top_players_scoring_diff(
        scoring_data, contest_style
    )

    lchv_count = _slate_overperformances(slate_id, service, fca, scoring_data)

    scoring = SlateScoreItem(
        top_possible_lineup_score=top_lineup_score,
        top_rational_lineup_score=brl_score,
        low_cost_high_value_player_count=lchv_count,
        top_players_scoring_diff_n=top_n_players_diff,
        top_players_scoring_diff_pctl=top_pctl_players_diff,
    )
    if score_cache is not None:
        score_cache[slate_id] = scoring
    return scoring


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the best lineup calculation")
    parser.add_argument("db_filename", help="database filename")
    parser.add_argument("service", help="service abbreviation")
    parser.add_argument("slate_id", help="slate id", type=int)

    _args = parser.parse_args()

    best_info = slate_scoring(_args.db_filename, _args.service, _args.slate_id)
    print(f"{best_info=}")
