#!/bin/bash

usage()
{
    echo "NBA Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
QB|WT|RB|D - Position
"
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
SEASONS="20192018 20182017 20172016 20162015"
DB="nba_hist.db"
MODEL=$1
SERVICE=$2

# total cases 20500
MAX_CASES=13500
MAX_OLS_FEATURES=61
source ${script_dir}/env.sc

if [ "$?" -eq 1 ] ||
       [ "$SERVICE" != "dk" -a "$SERVICE" != "fd" -a "$SERVICE" != "y" ]; then
    usage
    exit 1
fi

CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")

exit 1

PLAYER_STATS="p_bb p_cg p_er p_hbp p_hits
                  p_hr p_ibb p_ip p_k p_loss p_pc
                  p_qs p_runs p_strikes p_win p_wp"

TEAM_STATS="errors off_runs p_cg p_hold p_pc p_qs
                p_runs p_save win"

EXTRA_STATS="home_C opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
                   opp_starter_p_er opp_starter_p_loss opp_starter_p_qs opp_starter_p_runs
                   opp_starter_p_win player_home_H player_win team_home_H"

CUR_OPP_TEAM_STATS="off_1b off_2b off_3b off_bb off_hit
                        off_hr off_k off_pa off_rbi off_rbi_w2
                        off_rlob off_runs off_sac off_sb off_sb_c
                        p_er p_hold p_loss p_qs p_runs p_save
                        p_win win"

DATA_FILTER_FLAG="--mlb_only_starting_pitchers"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    exit 1
    EXTRA_STATS="$EXTRA_STATS venue_C"
fi


CMD="$CMD $DATA_FILTER_FLAG
-o nfl_${SERVICE}_${P_TYPE}_${MODEL}
${DB}
${CALC_ARGS}
--player_pos $POSITIONS
--player_stats $PLAYER_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--team_stats $TEAM_STATS
--extra_stats $EXTRA_STATS
--model_player_stat ${SERVICE}_score#
"

echo $CMD
