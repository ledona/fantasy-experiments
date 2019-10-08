#!/bin/bash

usage()
{
    echo "NFL team score prediction meval.
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

# set environment variables needed for analysis
script_dir="$(dirname "$0")"

MODEL=$1
DB="nfl_hist_2008-2018.scored.db"
SEASONS="2018 2017 2016 2015 2014"

# MAKE SURE THIS IS ACCURATE OR HIGHER
MAX_OLS_FEATURES=70
MAX_CASES=16000
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$2") && CMD=$(get_meval_base_cmd "$2")
ERROR=$?

if [ "$ERROR" -eq 1 ]; then
    usage
    exit 1
fi

exit 1
TEAM_STATS="win off_1b off_2b off_3b off_ab off_bb off_hbp
            off_hit off_hr off_k off_pa off_rbi
            off_rbi_w2 off_rlob off_runs off_sac off_sac_f
            off_sac_h off_sb off_sb_c"
    ('yds', "total yards of offense"),
    ('passing_yds', "passing yards"),
    ('rushing_yds', "rushing yards"),
    ('pts', "points scored"),
    ('turnovers', "turnovers recovered by other team"),
    ('op_yds', "yards allowed"),
    ('op_pts', "points allowed"),
    ('def_sacks', "sacks"),
    ('def_fumble_recov', "defensive fumble recoveries"),
    ('def_int', "defensive interceptions"),
    ('pens', 'number of penalties'),
    ('pen_yds', 'yards penalized'),
    ('win', 'team win=1, loss=0')

CUR_OPP_TEAM_STATS="errors p_bb p_cg p_er p_hbp p_hits p_hold p_hr p_ibb p_ip p_k
                    p_loss p_pc p_qs p_runs p_save p_strikes"
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
    

EXTRA_STATS="modeled_stat_trend modeled_stat_std_mean
             home_C l_hit_%_C l_hit_%_H
             opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
             opp_starter_p_bb opp_starter_p_cg opp_starter_p_er opp_starter_p_hbp
             opp_starter_p_hits opp_starter_p_hr opp_starter_p_ibb opp_starter_p_ip
             opp_starter_p_k opp_starter_p_loss opp_starter_p_pc opp_starter_p_qs
             opp_starter_p_runs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp
             r_hit_%_C r_hit_%_H
             team_home_H"

home_C - current home game status: 1 = home game, 0 = away game
modeled_stat_std_mean - Season to date mean for modeled stat
modeled_stat_trend - Value from (-1 - 1) describing the recent trend of the modeled value (similar to its slope)
        team_home_H - past home game status for a team (or a players current team): 1 = home game, 0 = away game

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C
                 opp_starter_phand_C opp_starter_phand_H
                 starter_phand_C"
fi

CMD="$CMD
-o nfl_team-score_${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat off_runs
"

echo $CMD
