#!/bin/bash

script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "LOL team prediction meval.
usage: $(basename "$0") ($AVAILABLE_MODELS) (w|dk|fd) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
SEASONS="2014 2015 2016 2017 2018 2019"
DB="lol_hist_2014-2019.scored.db"
MODEL=$1
TARGET=$2

if [ "$TARGET" == 'dk' -o "$TARGET" == 'fd' ]; then
    TARGET_STAT=${TARGET}_performance_score#
elif [ "$TARGET" == 'w' ]; then
    TARGET_STAT=w
else
    echo "Target of '${TARGET}' is not supoorted! Valid targets are w|dk|fd"
    exit 1
fi

# total cases 21740
MAX_CASES=20000
MAX_OLS_FEATURES=52
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")
ERROR=$?

if [ "$ERROR" -eq 1 ]; then
    usage
    exit 1
fi

TEAM_STATS="brnk
csat10 csat15 csdat10 csdat15 cspm
d dmgpm dpm drgk
gat10 gat15 gdat10 gdat15 gepm
k kpm
tk
vspm
wdcpm wdpm
xpat10 xpat15 xpdat10 xpdat15"

CUR_OPP_TEAM_STATS=$TEAM_STATS

EXTRA_STATS="
    modeled_stat_std_mean
    modeled_stat_trend
"

CMD="$CMD
-o lol-team-${MODEL}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat ${TARGET_STAT}
"

echo $CMD
