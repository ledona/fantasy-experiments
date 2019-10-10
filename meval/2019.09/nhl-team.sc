#!/bin/bash

usage()
{
    echo "NHL team score prediction meval.
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 1 ]; then
    usage
    echo Required arguments missing
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"

MODEL=$1
DB="nhl_hist_2008-2018.scored.db"
SEASONS="20192018 20182017 20172016 20162015"

# MAKE SURE THIS IS ACCURATE OR HIGHER
MAX_OLS_FEATURES=70
MAX_CASES=16000
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$2") && CMD=$(get_meval_base_cmd "$2")
ERROR=$?

if [ "$ERROR" -eq 1 ]; then
    usage
    echo Error parsing command line
    exit 1
fi


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

CUR_OPP_TEAM_STATS="
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

EXTRA_STATS="
home_C
modeled_stat_std_mean
modeled_stat_trend
player_home_H
player_pos_C
player_win
team_home_H
"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C
                 opp_starter_phand_C opp_starter_phand_H
                 starter_phand_C"
fi

CMD="$CMD
-o nhl_team-score_${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat off_runs
"

echo $CMD