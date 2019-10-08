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
SEASONS="20192018 20182017 20172016 20162015"
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

case $P_TYPE in
    G)
        POSITIONS="G"

        PLAYER_STATS="p_bb p_cg p_er p_hbp p_hits
                  p_hr p_ibb p_ip p_k p_loss p_pc
                  p_qs p_runs p_strikes p_win p_wp"
        ("toi_g", "time on ice for goalie (seconds)"),
    ("win", "on ice when game winning goal was scored, 1|0"),
    ("loss", "on ice when game lossing goal was scored, 1|0"),
    ("goal_ag", "goals against (regulation or overtime)"),
    ("save", "saves")
        
        TEAM_STATS="errors off_runs p_cg p_hold p_pc p_qs
                p_runs p_save win"

        EXTRA_STATS="home_C opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
                   opp_starter_p_er opp_starter_p_loss opp_starter_p_qs opp_starter_p_runs
                   opp_starter_p_win player_home_H player_win team_home_H"
        home_C - current home game status: 1 = home game, 0 = away game
        modeled_stat_std_mean - Season to date mean for modeled stat
        modeled_stat_trend - Value from (-1 - 1) describing the recent trend of the modeled value (similar to its slope)
        player_home_H - past home game status for a player: 1 = home game, 0 = away game
        player_pos_C - player position abbr for the case game
        player_win - Undefined for teams, for players these are recent wins for games they played in. 1 = win, .5 = tie, 0 = loss
        
        CUR_OPP_TEAM_STATS="off_1b off_2b off_3b off_bb off_hit
                        off_hr off_k off_pa off_rbi off_rbi_w2
                        off_rlob off_runs off_sac off_sb off_sb_c
                        p_er p_hold p_loss p_qs p_runs p_save
                        p_win win"

    ('ot', "won or lost in overtime 1|0"),
    ('so', "won or lost in shootout 1|0"),
    ("goal", "goals for"),
    ("goal_ag", "goals against"),
    ("save", "saves"),
    ("fo", "faceoffs"),
    ("fo_win_pct", "faceoff win %"),
    ("pp", "powerplays"),
    ("goal_pp", "powerplay goals"),
    ("pk", "penalty kills"),
    ("goal_pk_ag", "penalty kill goals against"),
    ("goal_sh", "shorthanded goals for"),
    ("goal_sh_ag", "shorthanded goals against"),
    ("shot", "shots"),
    ("shot_ag", "shots against"),
    ("pen", "Penalties"),
    ("pen_min", "penalty minutes"),
    ("hit", "hits"),

    # partially supported (not all seasons) by mysportsfeed, full support in nhlapi
    ("shot_b", "blocked shots"),

    # not supported by mysportsfeed, supported by nhlapi
    ("takeaway", "team takeaways"),
    ("giveaway", "team giveaways"),

    ('win', 'team win=1, loss=0')
        ;;
    S|CW|D)
        if [ "$P_TYPE" == "S" ]; then
            POSITIONS="LW RW C D"
        elif [ "$P_TYPE" == "CW" ]; then
            POSITIONS="LW RW C"
        elif [ "$P_TYPE" == "D" ]; then
            POSITIONS="D"
        else
            usage()
            echo Unhandled position ${P_TYPE}
        fi

        PLAYER_STATS="p_bb p_cg p_er p_hbp p_hits
                  p_hr p_ibb p_ip p_k p_loss p_pc
                  p_qs p_runs p_strikes p_win p_wp"
    ('goal', "goals"),
    ("assist", "assists"),
    ("pen", "penalties"),
    ("pen_mins", "penalty minutes"),
    ("goal_pp", "powerplay goals"),
    ("assist_pp", "powerplay assists"),
    ("goal_sh", "shorthanded goals"),
    ("assist_sh", "shorthanded assists"),
    ("goal_w", "game winning goals"),
    ("goal_t", "game tying goals"),
    ("goal_so", "shootout goals"),
    ("toi_ev", "even time on ice (seconds)"),
    ("toi_pp", "powerplay time on ice (seconds)"),
    ("toi_sh", "shorthanded time on ice (seconds)"),
    ("takeaway", "takeaways"),
    ("giveaway", "giveaways"),

    # non goalie stats
    ("fo", "faceoffs"),
    ("fo_win_pct", "faceoff win %"),
    ("hit", "hits"),
    ("pm", "plusminus"),
    ("shot", "shots on goal"),
    ("shot_b", "blocked shots"),
    ("line", "starting line (for goalie 1 represents the starting goalie)"),

        TEAM_STATS="errors off_runs p_cg p_hold p_pc p_qs
                p_runs p_save win"
    ('ot', "won or lost in overtime 1|0"),
    ('so', "won or lost in shootout 1|0"),
    ("goal", "goals for"),
    ("goal_ag", "goals against"),
    ("save", "saves"),
    ("fo", "faceoffs"),
    ("fo_win_pct", "faceoff win %"),
    ("pp", "powerplays"),
    ("goal_pp", "powerplay goals"),
    ("pk", "penalty kills"),
    ("goal_pk_ag", "penalty kill goals against"),
    ("goal_sh", "shorthanded goals for"),
    ("goal_sh_ag", "shorthanded goals against"),
    ("shot", "shots"),
    ("shot_ag", "shots against"),
    ("pen", "Penalties"),
    ("pen_min", "penalty minutes"),
    ("hit", "hits"),

    # partially supported (not all seasons) by mysportsfeed, full support in nhlapi
    ("shot_b", "blocked shots"),

    # not supported by mysportsfeed, supported by nhlapi
    ("takeaway", "team takeaways"),
    ("giveaway", "team giveaways"),

    ('win', 'team win=1, loss=0')

        EXTRA_STATS="home_C opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
                   opp_starter_p_er opp_starter_p_loss opp_starter_p_qs opp_starter_p_runs
                   opp_starter_p_win player_home_H player_win team_home_H"

        home_C - current home game status: 1 = home game, 0 = away game
        modeled_stat_std_mean - Season to date mean for modeled stat
        modeled_stat_trend - Value from (-1 - 1) describing the recent trend of the modeled value (similar to its slope)
        player_home_H - past home game status for a player: 1 = home game, 0 = away game
        player_pos_C - player position abbr for the case game
        player_win - Undefined for teams, for players these are recent wins for games they played in. 1 = win, .5 = tie, 0 = loss

        CUR_OPP_TEAM_STATS="off_1b off_2b off_3b off_bb off_hit
                        off_hr off_k off_pa off_rbi off_rbi_w2
                        off_rlob off_runs off_sac off_sb off_sb_c
                        p_er p_hold p_loss p_qs p_runs p_save
                        p_win win"
    ('ot', "won or lost in overtime 1|0"),
    ('so', "won or lost in shootout 1|0"),
    ("goal", "goals for"),
    ("goal_ag", "goals against"),
    ("save", "saves"),
    ("fo", "faceoffs"),
    ("fo_win_pct", "faceoff win %"),
    ("pp", "powerplays"),
    ("goal_pp", "powerplay goals"),
    ("pk", "penalty kills"),
    ("goal_pk_ag", "penalty kill goals against"),
    ("goal_sh", "shorthanded goals for"),
    ("goal_sh_ag", "shorthanded goals against"),
    ("shot", "shots"),
    ("shot_ag", "shots against"),
    ("pen", "Penalties"),
    ("pen_min", "penalty minutes"),
    ("hit", "hits"),

    # partially supported (not all seasons) by mysportsfeed, full support in nhlapi
    ("shot_b", "blocked shots"),

    # not supported by mysportsfeed, supported by nhlapi
    ("takeaway", "team takeaways"),
    ("giveaway", "team giveaways"),

    ('win', 'team win=1, loss=0')
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
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--team_stats $TEAM_STATS
--extra_stats $EXTRA_STATS
--model_player_stat ${SERVICE}_score#
"

echo $CMD
