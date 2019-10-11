#!/bin/bash

usage()
{
    echo "NBA Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y) [--test]

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
SEASONS="20182019 20172018 20162017 20152016 20142015"
DB="nba_hist_20082009-20182019.scored.db"
MODEL=$1
SERVICE=$2

# total cases 124404
MAX_CASES=80000
MAX_OLS_FEATURES=65
source ${script_dir}/env.sc

if [ "$?" -eq 1 ] ||
       [ "$SERVICE" != "dk" -a "$SERVICE" != "fd" -a "$SERVICE" != "y" ]; then
    usage
    echo Error parsing args or getting shared settings
    exit 1
fi

CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")

PLAYER_STATS="
    starter
    time
    pts
    asst
    turnovers
    stls
    blks
    fg_att
    fg_made
    tfg_att
    tfg_made
    ft_att
    ft_made
    d_reb
    o_reb
    fouls
    pm
    "

TEAM_STATS="
    pts
    asst
    turnovers
    fg_att
    fg_made
    tfg_att
    tfg_made
    ft_att
    ft_made
    o_reb
    op_pts
    op_fg_att
    op_fg_made
    op_tfg_att
    op_tfg_made
    op_ft_att
    op_o_reb
    stls
    blks
    d_reb
    fouls
    win
    "

EXTRA_STATS="
home_C
modeled_stat_std_mean
modeled_stat_trend
team_home_H
"

CUR_OPP_TEAM_STATS="
    pts
    turnovers
    fg_att
    fg_made
    tfg_att
    tfg_made
    ft_att
    o_reb
    op_pts
    op_asst
    op_fg_att
    op_fg_made
    op_tfg_att
    op_tfg_made
    op_ft_att
    op_o_reb
    stls
    blks
    d_reb
    fouls
    win
"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C"
fi


CMD="$CMD $DATA_FILTER_FLAG
-o nfl_${SERVICE}_${P_TYPE}_${MODEL}
${DB}
${CALC_ARGS}
--player_stats $PLAYER_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--team_stats $TEAM_STATS
--extra_stats $EXTRA_STATS
--model_player_stat ${SERVICE}_score#
"

echo $CMD
