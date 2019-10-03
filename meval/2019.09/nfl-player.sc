#!/bin/bash

usage()
{
    echo "NFL Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (QB|WT|WR|TE|RB|D) (dk|fd|y) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
QB|WT|WR|TE|RB|D - Position
"
}

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
MODEL=$1
P_TYPE=$2
SERVICE=$3
SEASONS="2018 2017 2016 2015 2014"
DB="nfl_hist.db"

if [ "$P_TYPE" != "" ]; then
    case $P_TYPE in
        QB|D)
            # total cases 20500
            MAX_CASES=13500
            MAX_OLS_FEATURES=61
            ;;
        WT)
            # both WR and TE
            # total cases 20500
            MAX_CASES=13500
            MAX_OLS_FEATURES=61
            ;;
        WR)
            # total cases 20500
            MAX_CASES=13500
            MAX_OLS_FEATURES=61
            ;;
        TE)
            # total cases 20500
            MAX_CASES=13500
            MAX_OLS_FEATURES=61
            ;;
        RB)
            # total cases 20500
            MAX_CASES=13500
            MAX_OLS_FEATURES=61
            ;;
        *)
            usage
            exit 1
    esac

    source ${script_dir}/env.sc
fi

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
    QB)
        POSITIONS="QB"
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
        ;;
    WT|TE|WR)
        # wide rceiver tight end
        POSITIONS="WR TE"
        exit 1

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
    RB)
        # wide rceiver tight end
        POSITIONS="RB"
        exit 1
        ;;
    D)
        # wide rceiver tight end
        POSITIONS="DEF"
        exit 1
        ;;
    *)
        echo Unhandled position $P_TYPE
        exit 1
        ;;
esac

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C"

    if [ "$P_TYPE" == "D" ]; then
        # defensive extras
        exit 1
        EXTRA_STATS="$EXTRA_STATS off_hit_side player_pos_C
                     opp_starter_phand_C opp_starter_phand_H"
    else
        # player extras
        exit 1
    fi
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
