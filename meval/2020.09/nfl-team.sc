#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "NFL team score prediction meval.
usage: $(basename "$0") ($AVAILABLE_MODELS) (win|pts) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

MODEL=$1
DB="nfl_hist_2009-2019.scored.db"
# evaluation against 2018
SEASONS="2019 2017 2016 2015 2014 2013 2012 2011"
OUT_STAT=$2
if [ $OUT_STAT != 'win' -a $OUT_STAT != 'pts' ]; then
    echo "Output stat '$OUT_STAT' is not valid!"
    exit 1
fi

# total cases 3858
MAX_OLS_FEATURES=31
MAX_CASES=3500

source ${script_dir}/env.sc


# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")
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
--model_team_stat $OUT_STAT
"

echo $CMD
