#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/mlb-env.sc

usage()
{
    echo "MLB Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (P|H) (dk|fd|y)"
}

CALC=CALC_${1}
P_TYPE=$2
SERVICE=$3

if [ -z "${!CALC}" ] ||
       [ "$SERVICE" != "dk" -a "$SERVICE" != "fd" -a "$SERVICE" != "y" ] ||
       [ "$P_TYPE" != "P" -a "$P_TYPE" != "H" ]; then
    usage
    exit 1
fi


if [ "$P_TYPE" == "P" ]; then
    # pitcher stuff
    POSITIONS="P"

    PLAYER_STATS="p_bb p_cg p_er p_hbp p_hits
                  p_hr p_ibb p_ip p_k p_loss p_pc
                  p_qs p_runs p_strikes p_win p_wp"

    TEAM_STATS="errors off_runs p_cg p_hold p_pc p_qs
                p_runs p_save win"

    SPECIAL_STATS="home_C opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
                   opp_starter_p_er opp_starter_p_loss opp_starter_p_qs opp_starter_p_runs
                   opp_starter_p_win player_home_H player_win team_home_H
                   venue_C venue_H"
elif [ "$P_TYPE" == "H" ]; then
    # hitter stuff
    POSITIONS="LF CF RF 1B 2B 3B SS C"

    PLAYER_STATS="off_1b off_2b off_3b off_bb off_bo
                  off_hbp off_hit off_hr off_k off_pa
                  off_rbi off_rbi_w2 off_rlob off_runs
                  off_sac off_sb off_sb_c"

    TEAM_STATS="off_1b off_2b off_3b off_bb
                off_hit off_hr off_k off_pa
                off_rbi off_rbi_w2 off_rlob off_runs
                off_sac off_sb off_sb_c p_runs win"


    SPECIAL_STATS="home_C off_hit_side
                   opp_starter_p_bb opp_starter_p_cg opp_starter_p_er opp_starter_p_hbp
                   opp_starter_p_hits opp_starter_p_hr opp_starter_p_ibb opp_starter_p_ip
                   opp_starter_p_k opp_starter_p_loss opp_starter_p_pc opp_starter_p_qs
                   opp_starter_p_runs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp
                   opp_starter_phand_C opp_starter_phand_H
                   player_home_H player_pos_C team_home_H venue_C venue_H"
fi



CMD="python -O scripts/meval.sc
$MEVAL_ARGS $STATS
--player_stats $PLAYER_STATS
--team_stats $TEAM_STATS
--special_stats $SPECIAL_STATS
-o mlb_${SERVICE}_${P_TYPE}
--player_pos $POSITIONS
mlb_hist_2008-2018.scored.db ${!CALC}
--model_player_stat ${SERVICE}_score#"

echo $CMD
