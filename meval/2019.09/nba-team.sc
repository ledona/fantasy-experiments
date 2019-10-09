#!/bin/bash

usage()
{
    echo "NBA team score prediction meval.
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

# set environment variables needed for analysis
script_dir="$(dirname "$0")"

MODEL=$1
DB="nba_hist_2008-2018.scored.db"
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
    exit 1
fi


TEAM_STATS="
    pts
    asst
    turnovers
    fg_att
    fg_made
    tfg_att
    tfg_made
    ft_att
    ft_made
    o_reb
    op_pts
    op_fg_att
    op_fg_made
    op_tfg_att
    op_tfg_made
    op_ft_att
    op_o_reb
    stls
    blks
    d_reb
    fouls
    win
    "

CUR_OPP_TEAM_STATS="
    pts
    asst
    turnovers
    fg_att
    fg_made
    tfg_att
    tfg_made
    ft_att
    ft_made
    o_reb
    op_pts
    op_asst
    op_fg_att
    op_fg_made
    op_tfg_att
    op_tfg_made
    op_ft_att
    op_ft_made
    op_o_reb
    stls
    blks
    d_reb
    fouls
    win
"

EXTRA_STATS="
home_C - current home game status: 1 = home game, 0 = away game
modeled_stat_std_mean - Season to date mean for modeled stat
modeled_stat_trend - Value from (-1 - 1) describing the recent trend of the modeled value (similar to its slope)
        team_home_H - past home game status for a team (or a players current team): 1 = home game, 0 = away game
"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C
                 opp_starter_phand_C opp_starter_phand_H
                 starter_phand_C"
fi

CMD="$CMD
-o nba_team-score_${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat off_runs
"

echo $CMD
