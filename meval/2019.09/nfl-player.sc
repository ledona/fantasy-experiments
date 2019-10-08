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
SEASONS="2018 2017 2016 2015 2014"
DB="nfl_hist.db"

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
    QB)
        POSITIONS="QB"

        PLAYER_STATS=
    ('tds', "total touchdowns"),
    ('fumbles_lost', "fumbles recovered by the defense"),

    # passing (i.e. quarterback)
    ('passing_att', "attempted passes"),
    ('passing_cmp', "completed passes"),
    ('passing_ints', "interceptions thrown"),
    ('passing_tds', "touchdowns from passes"),
    ('passing_yds', "yards on passes"),
    ('passing_twoptm', "successful 2 pt passes"),

    # rushing
    ('rushing_att', "run attempts"),
    ('rushing_tds', "tds on runs"),
    ('rushing_yds', "yards on runs"),
    ('rushing_twoptm', "successful 2 pt rushes"),


        TEAM_STATS=
    ('yds', "total yards of offense"),
    ('passing_yds', "passing yards"),
    ('rushing_yds', "rushing yards"),
    ('pts', "points scored"),
    ('turnovers', "turnovers recovered by other team"),

    ('op_yds', "yards allowed"),
    ('op_pts', "points allowed"),
    ('op_turnovers', "turnovers recovered by other team"),
    ('def_fumble_recov', "defensive fumble recoveries"),
    ('def_int', "defensive interceptions"),
    ('pens', 'number of penalties'),
    ('pen_yds', 'yards penalized'),
    ('win', 'team win=1, loss=0')

        EXTRA_STATS="home_C opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
                   opp_starter_p_er opp_starter_p_loss opp_starter_p_qs opp_starter_p_runs
                   opp_starter_p_win player_home_H player_win team_home_H"

        CUR_OPP_TEAM_STATS=
    ('yds', "total yards of offense"),
    ('pts', "points scored"),
    ('turnovers', "turnovers recovered by other team"),

    ('op_yds', "yards allowed"),
    ('op_passing_yds', "passing yards allowed"),
    ('op_rushing_yds', "rushing yards allowed"),
    ('op_pts', "points allowed"),
    ('op_turnovers', "turnovers recovered by other team"),
    ('def_sacks', "sacks"),
    ('def_fumble_recov', "defensive fumble recoveries"),
    ('def_int', "defensive interceptions"),
    ('pens', 'number of penalties'),
    ('pen_yds', 'yards penalized'),
    ('win', 'team win=1, loss=0')

        ;;
    WT|TE|WR)
        # wide rceiver tight end
        if [ "$P_TYPE" == "WT" ]; then
            POSITIONS="WR TE"
        else
            # either WR or TE
            POSITIONS=$P_TYPE
        fi

        PLAYER_STATS=
    # offense misc
    ('tds', "total touchdowns"),
    ('fumbles_lost', "fumbles recovered by the defense"),

        TEAM_STATS=
    ('yds', "total yards of offense"),
    ('passing_yds', "passing yards"),
    ('rushing_yds', "rushing yards"),
    ('pts', "points scored"),
    ('turnovers', "turnovers recovered by other team"),

    ('op_yds', "yards allowed"),
    ('op_pts', "points allowed"),
    ('op_turnovers', "turnovers recovered by other team"),
    ('def_fumble_recov', "defensive fumble recoveries"),
    ('def_int', "defensive interceptions"),
    ('pens', 'number of penalties'),
    ('pen_yds', 'yards penalized'),
    ('win', 'team win=1, loss=0')


        EXTRA_STATS="modeled_stat_trend modeled_stat_std_mean home_C
                 opp_starter_p_bb opp_starter_p_cg opp_starter_p_er opp_starter_p_hbp
                 opp_starter_p_hits opp_starter_p_hr opp_starter_p_ibb opp_starter_p_ip
                 opp_starter_p_k opp_starter_p_loss opp_starter_p_pc opp_starter_p_qs
                 opp_starter_p_runs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp
                 player_home_H team_home_H"

        CUR_OPP_TEAM_STATS="errors p_bb p_cg p_er p_hbp p_hits
                        p_hold p_hr p_ibb p_k p_loss p_pc p_qs
                        p_runs p_save p_strikes p_win win"

        ;;
    RB)
        # wide rceiver tight end
        POSITIONS="RB"

        TEAM_STATS=
    ('yds', "total yards of offense"),
    ('passing_yds', "passing yards"),
    ('rushing_yds', "rushing yards"),
    ('pts', "points scored"),
    ('turnovers', "turnovers recovered by other team"),

    ('op_yds', "yards allowed"),
    ('op_pts', "points allowed"),
    ('op_turnovers', "turnovers recovered by other team"),
    ('def_fumble_recov', "defensive fumble recoveries"),
    ('def_int', "defensive interceptions"),
    ('pens', 'number of penalties'),
    ('pen_yds', 'yards penalized'),
    ('win', 'team win=1, loss=0')

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
