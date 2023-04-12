#!/bin/bash

usage()
{
    echo "NFL team score prediction meval.
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"

MODEL=$1
DB="nfl_hist_2009-2018.scored.db"
SEASONS="2018 2017 2016 2015 2014 2013 2012"

# MAKE SURE THIS IS ACCURATE OR HIGHER n=3326
MAX_OLS_FEATURES=31
MAX_CASES=2200
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$2") && CMD=$(get_meval_base_cmd "$2")
ERROR=$?

if [ "$ERROR" -eq 1 ]; then
    usage
    exit 1
fi

TEAM_STATS="
    yds
    passing_yds
    rushing_yds
    pts
    turnovers
    op_yds
    op_pts
    def_sacks
    def_fumble_recov
    def_int
    pens
    pen_yds
    win
    "

CUR_OPP_TEAM_STATS="
    yds
    pts
    turnovers
    op_yds
    op_passing_yds
    op_rushing_yds
    op_pts
    op_turnovers
    def_sacks
    def_fumble_recov
    def_int
    pens
    pen_yds
    win
    "

EXTRA_STATS="
    home_C
    modeled_stat_std_mean
    modeled_stat_trend
    team_home_H
"

CMD="$CMD
-o nfl_team-score_${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat pts
"

echo $CMD