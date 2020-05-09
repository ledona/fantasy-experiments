#!/bin/bash

usage()
{
    echo "LOL team prediction meval.
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (w|dk|fd) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
SEASONS="2016 2017 2018 2019"
DB="lol_hist_2016-2019.scored.db"
MODEL=$1
TARGET=$2

if [ "$TARGET" == 'dk' -o "$TARGET" == 'fd' ]; then
    TARGET_STAT=${TARGET}_score#
elif [ "$TARGET" == 'w' ]; then
    TARGET_STAT=w
else
    echo "Target of '${TARGET}' is not supoorted! Valid targets are w|dk|fd"
    exit 1
fi

# total cases 17782
MAX_CASES=11000
MAX_OLS_FEATURES=12
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")
ERROR=$?

if [ "$ERROR" -eq 1 ]; then
    usage
    exit 1
fi

TEAM_STATS="brnk drgk fb tk w"

CUR_OPP_TEAM_STATS="brnk drgk fb tk w"

EXTRA_STATS="
    modeled_stat_std_mean
    modeled_stat_trend
"

CMD="$CMD
-o lol-team_${MODEL}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat ${TARGET}
"

echo $CMD
