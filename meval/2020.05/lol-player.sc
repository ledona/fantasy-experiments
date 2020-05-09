#!/bin/bash

usage()
{
    echo "LOL Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 2 ]; then
    usage
    echo Missing required args
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
SEASONS="2016 2017 2018 2019"
DB="lol_hist_2016-2019.scored.db"
MODEL=$1
SERVICE=$2

# total cases 79220
MAX_CASES=52000
MAX_OLS_FEATURES=20
source ${script_dir}/env.sc

if [ "$?" -eq 1 ] ||
   [ "$SERVICE" != "dk" -a "$SERVICE" != "fd" ]; then
    usage
    echo Error parsing args or getting shared settings
    exit 1
fi

CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")

PLAYER_STATS="asst cs cspm d k"

TEAM_STATS="brnk drgk fb tk w"

EXTRA_STATS="
modeled_stat_std_mean
modeled_stat_trend
player_home_H
player_win
"

CUR_OPP_TEAM_STATS="brnk drgk fb tk w"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS player_pos_C"
fi


CMD="$CMD $DATA_FILTER_FLAG
-o lol-player_${MODEL}
${DB}
${CALC_ARGS}
--player_stats $PLAYER_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--team_stats $TEAM_STATS
--extra_stats $EXTRA_STATS
--model_player_stat ${SERVICE}_score#
"

echo $CMD
