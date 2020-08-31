#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "MLB Player meval
usage: $(basename "$0") ($AVAILABLE_MODELS) (P|H) (dk|fd|y) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
P|H    - Pitcher or Hitter modeling
"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

MODEL=$1
P_TYPE=$2
SERVICE=$3
DB="mlb_hist_2008-2019.scored.db"
# evaluations against 2018
SEASONS="2019 2017 2016 2015 2014"

case $P_TYPE in
    P)
        # total cases 20410
        MAX_CASES=20000
        MAX_OLS_FEATURES=61
        ;;
    H)
        # total cases 198241
        MAX_CASES=100000
        MAX_OLS_FEATURES=73
        ;;
    *)
        usage
        echo "Position ${P_TYPE} not recognized"
        exit 1
esac

source ${script_dir}/env.sc

if [ "$?" -eq 1 ] ||
       [ "$SERVICE" != "dk" -a "$SERVICE" != "fd" -a "$SERVICE" != "y" ]; then
    usage
    exit 1
fi

CALC_ARGS=$(get_calc_args "$MODEL" "$4") && CMD=$(get_meval_base_cmd "$4")
if [ "$?" -eq 1 ]; then
    usage
    exit 1
fi

case $P_TYPE in
    P)
        # pitcher stuff
        POSITIONS="P"

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
        ;;
    H)
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

        EXTRA_STATS="modeled_stat_trend modeled_stat_std_mean home_C
                 opp_starter_p_bb opp_starter_p_cg opp_starter_p_er opp_starter_p_hbp
                 opp_starter_p_hits opp_starter_p_hr opp_starter_p_ibb opp_starter_p_ip
                 opp_starter_p_k opp_starter_p_loss opp_starter_p_pc opp_starter_p_qs
                 opp_starter_p_runs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp
                 player_home_H team_home_H"

        CUR_OPP_TEAM_STATS="errors p_bb p_cg p_er p_hbp p_hits
                        p_hold p_hr p_ibb p_k p_loss p_pc p_qs
                        p_runs p_save p_strikes p_win win"

        DATA_FILTER_FLAG="--mlb_only_starting_hitters"
        ;;
    *)
        echo Unhandled position $P_TYPE
        exit 1
        ;;
esac

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C"

    if [ "$P_TYPE" == "H" ]; then
        # hitters get off hit side
        EXTRA_STATS="$EXTRA_STATS off_hit_side player_pos_C
                     opp_starter_phand_C opp_starter_phand_H"
    fi
fi


CMD="$CMD $DATA_FILTER_FLAG
-o mlb_${SERVICE}_${P_TYPE}_${MODEL}
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
