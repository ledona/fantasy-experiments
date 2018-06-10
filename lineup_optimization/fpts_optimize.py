"""
tune lineup params while optimizing for fpts using perfect data
"""
import argparse
import shlex
from ledona import process_timer
from dateutil.parser import parse

from fantasy_py.calculation.regression.util.hyper_search import (
    add_searcher_cmd_line_args, DOMAIN_SEARCHERS)
from fantasy_py.util import CLSRegistry


def create_hyper_param_dists(args):
    raise NotImplementedError()


def create_evaluation_function(args):
    raise NotImplementedError()


# TODO: move this to hyper_search
def process_searcher_args(args, hp_dists, game_date):
    """
    takes a Namespace/Attribute object with attributes definde in add_searcher_cmd_line_args
    and returns a searcher
    """
    raise NotImplementedError()


def run_evaluation(game_dates, searcher_args, hp_dists, eval_func,
                   verbose=False, progress=True):
    # iterate over games dates
    for date in game_dates:
        searcher = process_searcher_args(searcher_args, hp_dists, date)
        searcher.search(eval_func, desc_dict={'game_date': date})


@process_timer
def process_cmd_line(cmd_line_str=None):
    parser = argparse.ArgumentParser(description="tune lineup params while optimizing for fpts using perfect data")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--no_progress", default=True, dest="progress", action="store_false")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--generations_range", type=int, metavar=('min', 'max'), nargs=2)
    parser.add_argument("--population", type=int)
    parser.add_argument("--population_range", type=int, metavar=('min', 'max'), nargs=2)
    parser.add_argument("--offspring", type=int)
    parser.add_argument("--offspring_range", type=int, metavar=('min', 'max'), nargs=2)
    parser.add_argument("--crossover_prob", type=float, help="crossoverover probability")
    parser.add_argument("--crossover_prob_range", type=float, metavar=('min', 'max'), nargs=2)
    parser.add_argument("--mutation_prob", type=float, help="mutationover probability")
    parser.add_argument("--mutation_prob_range", type=float, metavar=('min', 'max'), nargs=2)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--runs_range", type=int, metavar=('min', 'max'), nargs=2)

    parser.add_argument("DB_FILE", help="The database file to match players and teams against")
    parser.add_argument("dates", nargs='+', type=parse,
                        help="game dates to use for evaluation")

    add_searcher_cmd_line_args(parser)

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    if args.resume:
        raise NotImplementedError()

    hp_dists = create_hyper_param_dists(args)
    searcher = process_searcher_args(args, hp_dists)
    eval_func = create_evaluation_function(args)

    run_evaluation(args.dates, searcher, eval_func,
                   verbose=args.verbose, progress=args.progress)


if __name__ == "__main__":
    process_cmd_line()
