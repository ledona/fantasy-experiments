#!/bin/bash

usage()
{
    echo "NFL Player meval
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (QB|WT|WR|TE|RB|D) (dk|fd|y) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
QB|WT|WR|TE|RB|D - Position
"
}

if [ "$#" -lt 3 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
MODEL=$1
P_TYPE=$2
SERVICE=$3
SEASONS="2018 2017 2016 2015 2014 2013 2012"
DB="nfl_hist_2009-2018.scored.db"

case $P_TYPE in
    QB)
        # total cases 3326
        MAX_CASES=2200
        MAX_OLS_FEATURES=44
        ;;
    D)
        # total cases 3326
        MAX_CASES=2200
        MAX_OLS_FEATURES=33
        ;;
    WT)
        # both WR and TE
        # total cases 16239
        MAX_CASES=10500
        MAX_OLS_FEATURES=28
        ;;
    WR)
        # total cases 10951
        MAX_CASES=7000
        MAX_OLS_FEATURES=28
        ;;
    TE)
        # total cases 5288
        MAX_CASES=3500
        MAX_OLS_FEATURES=28
        ;;
    RB)
        # total cases 7813
        MAX_CASES=5200
        MAX_OLS_FEATURES=32
        ;;
    K)
        # total cases 3326
        MAX_CASES=2000
        MAX_OLS_FEATURES=25
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

# team stats used for all modeling
SHARED_TEAM_STATS="
        yds
        pts
        turnovers
        op_yds
        op_pts
        op_turnovers
        def_fumble_recov
        def_int
        pens
        pen_yds
        win
        "

# team stats used for all player modeling
PLAYER_TEAM_STATS="${SHARED_TEAM_STATS}
    passing_yds
    rushing_yds
"

# opp team stats for all modeling AND team stats for a defense
CUR_OPP_TEAM_STATS="${SHARED_TEAM_STATS}
        op_passing_yds
        op_rushing_yds
        def_sacks
"

SHARED_EXTRA_STATS="
        home_C
        modeled_stat_std_mean
        modeled_stat_trend
"

PLAYER_EXTRA_STATS="${SHARED_EXTRA_STATS}
        player_home_H
        player_win
        "

case $P_TYPE in
    QB)
        POSITIONS="QB"

        PLAYER_STATS="
            tds
            fumbles_lost
            passing_att
            passing_cmp
            passing_ints
            passing_tds
            passing_yds
            passing_twoptm
            rushing_att
            rushing_tds
            rushing_yds
            rushing_twoptm
            "

        TEAM_STATS=$PLAYER_TEAM_STATS
        EXTRA_STATS=$PLAYER_EXTRA_STATS
        ;;
    WT|TE|WR)
        # wide rceiver tight end
        if [ "$P_TYPE" == "WT" ]; then
            POSITIONS="WR TE"
        else
            # either WR or TE
            POSITIONS=$P_TYPE
        fi

        PLAYER_STATS="
        tds
        fumbles_lost
        receiving_rec
        receiving_targets
        receiving_tds
        receiving_yds
        receiving_twoptm
        "

        TEAM_STATS=$PLAYER_TEAM_STATS
        EXTRA_STATS="$PLAYER_EXTRA_STATS player_pos_C"
        ;;
    RB)
        # wide rceiver tight end
        POSITIONS="RB"

        PLAYER_STATS="
        tds
        fumbles_lost
        rushing_att
        rushing_tds
        rushing_yds
        rushing_twoptm
        receiving_rec
        receiving_targets
        receiving_tds
        receiving_yds
        receiving_twoptm
        "

        TEAM_STATS=$PLAYER_TEAM_STATS
        EXTRA_STATS=$PLAYER_EXTRA_STATS
        ;;
    K)
        POSITIONS="K"
        PLAYER_STATS="
            kicking_xpa
            kicking_xpm
            kicking_fga
            kicking_fgm
            kicking_fgm_0-39
            kicking_fgm_40-49
            kicking_fgm_50+
            "
        TEAM_STATS=$SHARED_TEAM_STATS
        CUR_OPP_TEAM_STATS=$SHARED_TEAM_STATS
        EXTRA_STATS=$PLAYER_EXTRA_STATS
        ;;
    D)
        # wide rceiver tight end
        POSITIONS="DEF"

        TEAM_STATS=$CUR_OPP_TEAM_STATS

        EXTRA_STATS="${SHARED_EXTRA_STATS} team_home_H"
        ;;
    *)
        echo Unhandled position $P_TYPE
        exit 1
        ;;
esac


CMD="$CMD $DATA_FILTER_FLAG
-o nfl_${SERVICE}_${P_TYPE}_${MODEL}
${DB}
${CALC_ARGS}
--player_pos $POSITIONS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--team_stats $TEAM_STATS
"

if [ "$PLAYER_STATS" != "" ]; then
    CMD="$CMD --player_stats $PLAYER_STATS
         --model_player_stat ${SERVICE}_score_off#"
else
    # team defense!
    CMD="$CMD --model_team_stat ${SERVICE}_score_def#"
fi

echo $CMD
