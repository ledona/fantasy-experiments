#!/bin/bash

usage()
{
    echo "NHL Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (G|S|CW|D) (dk|fd|y) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
G|S|CW|D - Position, S-All skaters, CW-Center and Forward, D - Defender
"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
SEASONS="20182019 20172018 20162017 20152016 20142015"
DB="nhl_hist.db"
MODEL=$1
P_TYPE=$2
SERVICE=$3

case $P_TYPE in
    G)
        # total cases 20500
        POSITIONS="G"
        MAX_CASES=13500
        MAX_OLS_FEATURES=61
        ;;
    S)
        # total cases 20500
        MAX_CASES=13500
        MAX_OLS_FEATURES=61
        ;;
    CW)
        # total cases 20500
        MAX_CASES=13500
        MAX_OLS_FEATURES=61
        ;;
    D)
        # total cases 20500
        MAX_CASES=13500
        MAX_OLS_FEATURES=61
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

# all positions use the same team stats
TEAM_STATS="
        ot
        so
        goal
        goal_ag
        save
        fo
        fo_win_pct
        pp
        goal_pp
        pk
        goal_pk_ag
        goal_sh
        goal_sh_ag
        shot
        shot_ag
        pen
        pen_min
        hit
        shot_b
        takeaway
        giveaway
        win
"

# all positions use the same extra stats
EXTRA_STATS="
        home_C
        modeled_stat_std_mean
        modeled_stat_trend
        player_home_H
        player_pos_C
        player_win
        "


case $P_TYPE in
    G)
        POSITIONS="G"

        PLAYER_STATS="
        toi_g
        win
        loss
        goal_ag
        save
        "

        DATA_FILTER_FLAG="--only_starting_goalies"
        ;;
    S|CW|D)
        if [ "$P_TYPE" == "S" ]; then
            POSITIONS="LW RW C D"
        elif [ "$P_TYPE" == "CW" ]; then
            POSITIONS="LW RW C"
        elif [ "$P_TYPE" == "D" ]; then
            POSITIONS="D"
        else
            usage
            echo "Unhandled position ${P_TYPE}"
        fi

        PLAYER_STATS="
        goal
        assist
        pen
        pen_mins
        goal_pp
        assist_pp
        goal_sh
        assist_sh
        goal_w
        goal_t
        goal_so
        toi_ev
        toi_pp
        toi_sh
        takeaway
        giveaway
        fo
        fo_win_pct
        hit
        pm
        shot
        shot_b
        line
        "
        ;;
    *)
        usage
        exit 1
esac

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C"

    if [ "$P_TYPE" == "G" ]; then
        # defensive extras
        exit 1
        EXTRA_STATS="$EXTRA_STATS off_hit_side player_pos_C
                     opp_starter_phand_C opp_starter_phand_H"
    else
        # skater extras
        exit 1
    fi
fi


CMD="$CMD $DATA_FILTER_FLAG
-o nfl_${SERVICE}_${P_TYPE}_${MODEL}
${DB}
${CALC_ARGS}
--player_pos $POSITIONS
--player_stats $PLAYER_STATS
--cur_opp_team_stats $TEAM_STATS
--team_stats $TEAM_STATS
--extra_stats $EXTRA_STATS
--model_player_stat ${SERVICE}_score#
"

echo $CMD
