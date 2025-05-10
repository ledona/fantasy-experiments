import argparse
import json
import math
import os
from argparse import Namespace
from contextlib import contextmanager
from typing import Literal, cast

import pandas as pd
from fantasy_py import FANTASY_SERVICE_DOMAIN, CLSRegistry, DataNotAvailableException, db, log
from fantasy_py.lineup import FantasyService, gen_lineups
from fantasy_py.lineup.do_gen_lineup import lineup_plan_helper
from fantasy_py.lineup.knapsack import MixedIntegerKnapsackSolver
from fantasy_py.sport import Starters

_LOGGER = log.get_logger(__name__)


TopScoreCacheMode = Literal["default", "overwrite", "missing"]


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


@contextmanager
def score_cache_ctx(sport: str, top_score_cache_mode: TopScoreCacheMode, cache_dir="."):
    """context manager for caching top and mean-diff-hist-vs-pred scores"""
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist")
    score_cache_filename = sport + "-slate.score.json"
    score_cache_filepath = os.path.join(cache_dir, score_cache_filename)
    score_dict: dict[int, tuple[float, float]]
    orig_score_dict = {}

    if os.path.isfile(score_cache_filepath):
        if top_score_cache_mode in ("default", "missing"):
            with open(score_cache_filepath, "r") as f:
                cache_data = json.load(f)
            for slate_id, score in cache_data.items():
                if top_score_cache_mode == "missing" and score is None:
                    continue
                orig_score_dict[int(slate_id)] = score
        elif top_score_cache_mode == "overwrite":
            _LOGGER.info("Overwriting existing best score cache data at '%s'", score_cache_filepath)
        else:
            raise ValueError("Unexpected top score cache mode", top_score_cache_mode)
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


_TOP_PLAYER_PCTL = 0.4
"""used to find how well top players performed on average vs expectations"""


def slate_scoring(
    db_filename,
    service_abbr,
    slate_id,
    score_cache: None | dict[int, None | tuple[float, float]] = None,
    sport: str | None = None,
    screen_lineup_constraints_mode="fail",
) -> tuple[None, None] | tuple[float, float]:
    """
    Calculate the best possible fantasy score and difference between mean historic score
    vs predicted scores for top players for the requested slate.

    Function is used as a map function for a pandas series.

    pts_stats_names - the statistic names for the scores to use for players/teams
    best_score_cache - cache of slate ids mapped to their score. this will be
        searched and possibly updated to include the score for the requested slate

    returns - None if there is an error occurs
    """
    if not isinstance(slate_id, (int, float)) or math.isnan(slate_id):
        return None, None

    slate_id = int(slate_id)
    if score_cache:
        if slate_id in score_cache:
            return score_cache[slate_id]
        _LOGGER.info("slate_id=%i not in best score cache", slate_id)

    db_obj = db.get_db_obj(db_filename)
    if sport:
        assert sport == db_obj.db_manager.ABBR
    else:
        sport = db_obj.db_manager.ABBR

    # slate date
    with db_obj.session_scoped() as session:
        slate = (
            session.query(db.DailyFantasySlate)
            .filter(db.DailyFantasySlate.id == int(slate_id))
            .one_or_none()
        )
        if slate is None:
            _LOGGER.warning("Error: Unable to find slate_id=%i in database", slate_id)
            return None, None

        game_date = slate.date
        slate_name = slate.name
        service = slate.service

    _LOGGER.info(
        "Generating best historic lineup for %s slate '%s' (%i)", game_date, slate_name, slate_id
    )

    # TODO: the following should also take slate_id
    # get the starters
    starters = db_obj.db_manager.get_starters(
        service,
        games_date=game_date,
        db_obj=db_obj,
        slate=slate_id,
    )
    if slate_name not in starters.slates:
        raise ValueError(
            f"{slate_name=} not in starters. Starters slates are {starters.slates.keys()}"
        )
    starters: Starters = starters.filter_by_slate(slate_name)

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
    args, fca = db_obj.db_manager.gen_lineups_preprocess(
        db_obj, args, None, game_date, starters=starters, print_slate_info=False
    )[:2]

    service_cls = cast(FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, service))

    args = lineup_plan_helper(args, db_obj, starters, service_cls, [], slate_name)[0]
    constraints = service_cls.get_constraints(
        db_obj.db_manager.ABBR, slate=starters.slates[slate_name]
    )
    assert constraints is not None

    solver = MixedIntegerKnapsackSolver(
        constraints.lineup_constraints,
        constraints.budget,
        totals_func=constraints.totals_func,
        fill_all_positions=constraints.fill_all_positions,
    )

    epoch = db_obj.db_manager.epoch_for_date(game_date)

    try:
        lineups, score_data = gen_lineups(
            db_obj,
            fca,
            service_cls.DEFAULT_MODEL_NAMES.get(sport),
            solver,
            service_cls,
            n_lineups=1,
            slate=slate_name,
            slate_info=starters.slates[slate_name],
            score_data_type="historic",
            slate_epoch=epoch,
            screen_lineup_constraints_mode=screen_lineup_constraints_mode,
            scores_to_include=["predicted"],
        )
        hist_score = cast(float, lineups[0].historic_fpts)
    except DataNotAvailableException as dna_ex:
        _LOGGER.warning(
            "Lineup generation data not available for service_abbr='%s' sport='%s' "
            "slate_id=%i on %s: %s",
            service_abbr,
            sport,
            slate_id,
            game_date,
            dna_ex,
        )
        return None, None
    except Exception as ex:
        _LOGGER.error(
            "Error calculating best lineup for service_abbr='%s' sport='%s' slate_id=%i on %s.",
            service_abbr,
            sport,
            slate_id,
            game_date,
            exc_info=ex,
        )
        raise

    top_predicted_players_scores = (
        pd.merge(
            score_data["historic"].rename(columns={"fpts": "hist-fpts"}),
            score_data["predicted"].rename(columns={"fpts": "pred-fpts"}),
            on=["game_id", "team_id", "player_id"],
            how="inner",
        )
        .sort_values("pred-fpts", ascending=False)
        .head(int(len(score_data["predicted"]) * _TOP_PLAYER_PCTL))
    )
    hist_pred_diff = float(
        top_predicted_players_scores["hist-fpts"].mean()
        - top_predicted_players_scores["pred-fpts"].mean()
    )
    if score_cache is not None and hist_score is not None:
        score_cache[slate_id] = hist_score, hist_pred_diff
    return hist_score, hist_pred_diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the best lineup calculation")
    parser.add_argument("db_filename", help="database filename")
    parser.add_argument("service", help="service abbreviation")
    parser.add_argument("slate_id", help="slate id", type=int)

    _args = parser.parse_args()

    best_info = slate_scoring(_args.db_filename, _args.service, _args.slate_id)
    print(f"{best_info=}")
