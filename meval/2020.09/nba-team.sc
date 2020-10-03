#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "NBA team score prediction meval.
usage: $(basename "$0") ($AVAILABLE_MODELS) (pts|win) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi


MODEL=$1
# eval against 20182019
SEASONS="20172018 20162017 20152016 20142015 20132014"
DB="nba_hist_20082009-20192020.precovid.scored.db"
OUT_STAT=$2

if [ "$OUT_STAT" != "win" -a "$OUT_STAT" != "pts" ]; then
    echo "Unrecognized output stat '$OUT_STAT'"
    exit 1
fi

# MAKE SURE THIS IS ACCURATE OR HIGHER n=12083
MAX_OLS_FEATURES=50
MAX_CASES=10000
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")
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
home_C
modeled_stat_std_mean
modeled_stat_trend
team_home_H
"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C"
fi

CMD="$CMD
-o nba_team_${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat $OUT_STAT
"

echo $CMD
