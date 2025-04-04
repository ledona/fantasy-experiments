import argparse
import json
import math
import os
from argparse import Namespace
from contextlib import contextmanager
from typing import Literal, cast

from fantasy_py import FANTASY_SERVICE_DOMAIN, CLSRegistry, db, log
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
def best_score_cache_ctx(sport: str, top_score_cache_mode: TopScoreCacheMode, cache_dir="."):
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist")
    top_score_cache_filename = sport + "-slate.top_score.json"
    top_score_cache_filepath = os.path.join(cache_dir, top_score_cache_filename)
    top_score_dict: dict[int, float]
    orig_top_score_dict = {}

    if os.path.isfile(top_score_cache_filepath):
        if top_score_cache_mode in ("default", "missing"):
            with open(top_score_cache_filepath, "r") as f:
                cache_data = json.load(f)
            for slate_id, score in cache_data.items():
                if top_score_cache_mode == "missing" and score is None:
                    continue
                orig_top_score_dict[int(slate_id)] = score
        elif top_score_cache_mode == "overwrite":
            _LOGGER.info(
                "Overwriting existing best score cache data at '%s'", top_score_cache_filepath
            )
        else:
            raise ValueError("Unexpected top score cache mode", top_score_cache_mode)
    else:
        _LOGGER.info("Best score cache data not found! '%s'", top_score_cache_filepath)
        orig_top_score_dict = {}

    # make a copy so that we can figure out if there are updates
    # TODO: for diff, can probably do this more efficiently by comparing a hash of the before and after
    top_score_dict = dict(orig_top_score_dict)

    try:
        yield top_score_dict
    finally:
        if orig_top_score_dict != top_score_dict:
            # TODO: should save the cache as new scores are added
            _LOGGER.info(
                "Writing updated best score values to cache '%s'", top_score_cache_filepath
            )
            with open(top_score_cache_filepath, "w") as f:
                json.dump(top_score_dict, f)
        _LOGGER.info("Exiting best_score_cache")


def best_possible_lineup_score(
    db_filename,
    service_abbr,
    slate_id,
    best_score_cache: None | dict[int, None | float] = None,
    sport: str | None = None,
    screen_lineup_constraints_mode="fail",
) -> None | float:
    """
    calculate the best possible fantasy score for the requested slate
    used as a map function for a pandas series.

    pts_stats_names - the statistic names for the scores to use for players/teams
    best_score_cache - cache of slate ids mapped to their score. this will be
        searched and possibly updated to include the score for the requested slate

    returns - None if there is an error calculating the best possible score
    """
    if not isinstance(slate_id, (int, float)) or math.isnan(slate_id):
        return None

    slate_id = int(slate_id)
    if best_score_cache:
        if slate_id in best_score_cache:
            return best_score_cache[slate_id]
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
            return None

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
        lineups = gen_lineups(
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
        )[0]
        hist_score = lineups[0].historic_fpts
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

    if best_score_cache is not None and hist_score is not None:
        best_score_cache[slate_id] = hist_score
    return hist_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the best lineup calculation")
    parser.add_argument("db_filename", help="database filename")
    parser.add_argument("service", help="service abbreviation")
    parser.add_argument("slate_id", help="slate id", type=int)

    _args = parser.parse_args()

    best_score = best_possible_lineup_score(_args.db_filename, _args.service, _args.slate_id)
    print(f"{best_score=}")
