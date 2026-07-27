import argparse
import json
import math
import os
from argparse import Namespace
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import astuple
from typing import Literal, cast

from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    DataNotAvailableException,
    UnexpectedValueError,
    db,
    log,
)
from fantasy_py.analysis.backtest.daily_fantasy import (
    SlateScoreItem,
    bt_addl_slate_data,
    bt_score_lineup,
    bt_slate_overperformances,
    bt_top_players_scoring_diff,
    get_best_rational_lineup,
    get_rational_lineup_gen_params,
)
from fantasy_py.lineup import FantasyService, GenLineupsParams
from fantasy_py.lineup.knapsack import MixedIntegerKnapsackSolver
from fantasy_py.sport import SportDBManager
from ledona import constant_hasher
from typeguard import check_type

_LOGGER = log.get_logger(__name__)


SlateScoreCacheMode = Literal["default", "overwrite", "missing"]


class ScoreCache:
    """cache for slate scores"""

    _UPDATE_TO_SAVE_PERIOD = 10
    """how many updates before the cache saves its current state"""

    def __init__(
        self,
        score_cache_filepath: str,
        rlp_hash,
        cache_mode: SlateScoreCacheMode = "default",
    ):
        self.data: dict[int, SlateScoreItem] = {}
        self._unsaved_updates = 0
        """changes that have been made since the data was loaded/last-saved"""

        self._rlp_hash = rlp_hash
        """hash used to test if cached data should be reused"""

        self.score_cache_filepath = score_cache_filepath

        if cache_mode == "overwrite":
            _LOGGER.info("Overwriting existing best score cache data at '%s'", score_cache_filepath)
            return

        if not os.path.isfile(score_cache_filepath):
            _LOGGER.info(
                "Best score cache data not found! Starting a new cache at '%s'",
                score_cache_filepath,
            )
            return

        with open(score_cache_filepath, "r") as f:
            cache_data = json.load(f)

        if cache_data.get("rational_lineup_params_hash") != rlp_hash:
            _LOGGER.warning(
                "Cached slate scores in '%s' cannot be used because of change in lineup gen params",
                score_cache_filepath,
            )
            return

        for slate_id, score_data in cache_data["scores"].items():
            if cache_mode == "missing" and score_data is None:
                continue
            try:
                self.data[int(slate_id)] = SlateScoreItem(*score_data)
            except TypeError:
                _LOGGER.error(
                    "Parsing error while loading slate score cache from '%s'. Cache will be rebuilt",
                    score_cache_filepath,
                )
                self.data = {}
                return

        _LOGGER.success(
            "Reusing %d cached slate score entries found at '%s'",
            len(self.data),
            score_cache_filepath,
        )

    def __setitem__(self, slate_id, item: SlateScoreItem):
        if self.data.get(slate_id) == item:
            return
        self.data[slate_id] = item
        self._unsaved_updates += 1
        if self._unsaved_updates >= self._UPDATE_TO_SAVE_PERIOD:
            self.save()

    def __getitem__(self, slate_id):
        return self.data.get(slate_id)

    def __contains__(self, slate_id):
        return slate_id in self.data

    def __len__(self):
        return len(self.data)

    def save(self):
        if not self._unsaved_updates > 0:
            return

        data = {slate_id: astuple(ssi) for slate_id, ssi in self.data.items()}
        with open(self.score_cache_filepath, "w") as f:
            json.dump(
                {"rational_lineup_params_hash": self._rlp_hash, "scores": data},
                f,
                indent=2,
            )
        _LOGGER.success(
            "Saved %d items (%d new items) to best score cache '%s'",
            len(self.data),
            self._unsaved_updates,
            self.score_cache_filepath,
        )

        self._unsaved_updates = 0


@contextmanager
def score_cache_ctx(sport: str, contest_style, cache_mode: SlateScoreCacheMode, cache_dir="."):
    """context manager for caching lineup scoring results"""
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Cache directory '{cache_dir}' does not exist")
    score_cache_filename = f"{sport}-{contest_style.value}-slate.score.json"
    score_cache_filepath = os.path.join(cache_dir, score_cache_filename)

    rational_lineup_params = get_rational_lineup_gen_params(sport, contest_style.value)
    rlp_hash = constant_hasher(rational_lineup_params)

    score_cache = ScoreCache(score_cache_filepath, rlp_hash, cache_mode)

    try:
        yield score_cache
    finally:
        score_cache.save()

        counts: dict[int, int] = defaultdict(int)
        game_counts: dict[int, list[int]] = defaultdict(list)
        for score in score_cache.data.values():
            idx = score.rational_lineup_settings_index + 1
            counts[idx] += 1
            game_counts[idx].append(score.games_count)
        rls_report = "\n".join(
            f"{i}\t{counts[i]}\t{min(game_counts[i])}-{max(game_counts[i])}"
            if game_counts[i]
            else f"{i}\t{counts[i]}\t-"
            for i in range(1, len(rational_lineup_params) + 1)
        )
        print(f"""

*** Rational Lineup Settings Usage {sport=} contest_style={contest_style.value} ***
params\tcount\tgames
{rls_report}
**************************************

""")
        _LOGGER.info("Exiting best_score_cache")


def slate_scoring(
    session,
    slate_id,
    score_cache: None | ScoreCache = None,
    screen_lineup_constraints_mode="fail",
):
    """
    Calculate the best possible fantasy score and difference between mean historic score
    vs predicted scores for top players for the requested slate.

    Function is used as a map function for a pandas series.

    pts_stats_names - the statistic names for the scores to use for players/teams
    score_cache - cache of slate ids mapped to their score. this will be
        searched and possibly updated to include the score for the requested slate

    returns - None if there is an error occurs, otherwise a tuple of
        (top-possible-lineup-score,
         top-possible-lineup-score - predicted-score-for-top-possible-lineup,
         top-rational-lineup-score,
         low-cost-high-value-player-count)
    """
    if not isinstance(slate_id, (int, float)) or math.isnan(slate_id):
        raise UnexpectedValueError("id of slate is expected to be a number")

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

    game_date = slate.date
    slate_name = slate.name
    service = slate.service

    _LOGGER.info(
        "Generating best historic lineups for %s slate '%s' (%i)", game_date, slate_name, slate_id
    )

    # get the starters
    starters = db_manager.get_starters(
        service, games_date=game_date, db_obj=session.info["db_obj"], slate=slate_name
    )
    if starters is None or starters.slates is None:
        raise DataNotAvailableException(
            f"Failed to retrieve starters for {game_date=} {slate_id=} {slate_name=}"
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

    top_lineup_solver = MixedIntegerKnapsackSolver(
        contest_constraints.knapsack_constraints,
        contest_constraints.budget,
        totals_func=contest_constraints.totals_func,
        fill_all_positions=contest_constraints.fill_all_positions,
    )

    top_lineup_params = GenLineupsParams(score_data_type="historic", n_lineups=1)

    lineup_score_info = bt_score_lineup(
        session,
        fca,
        top_lineup_solver,
        service_cls,
        slate_name,
        starters.slates[slate_name],
        screen_lineup_constraints_mode,
        "top-lineup",
        top_lineup_params,
        True,
    )
    if lineup_score_info is None:
        return None

    top_lineup, scoring_data = lineup_score_info
    top_lineup_score = check_type(top_lineup.historic_fpts, float)
    contest_style = str(slate.style)

    addl_scoring = bt_addl_slate_data(fca, top_lineup, "all")

    br_tries, br_lineup = get_best_rational_lineup(
        session, fca, slate_name, slate_info, service_cls, contest_constraints
    )
    br_lineup_pts = br_lineup.historic_fpts
    assert br_lineup_pts is not None

    top_pctl_players_diff, top_n_players_diff = bt_top_players_scoring_diff(
        scoring_data, contest_style
    )

    lchv_count = bt_slate_overperformances(slate_id, service, fca, scoring_data)

    scoring = SlateScoreItem(
        top_possible_lineup_score=top_lineup_score,
        top_rational_lineup_score=br_lineup_pts,
        rational_lineup_settings_index=br_tries - 1,
        low_cost_high_value_player_count=lchv_count,
        top_players_scoring_diff_n=top_n_players_diff,
        top_players_scoring_diff_pctl=top_pctl_players_diff,
        addl_scoring=addl_scoring,
        games_count=len(starters.games),
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
