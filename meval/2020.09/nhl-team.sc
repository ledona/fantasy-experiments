#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "NHL team score prediction meval.
usage: $(basename "$0") ($AVAILABLE_MODELS) (goal|win) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 2 ]; then
    usage
    echo Required arguments missing
    exit 1
fi


MODEL=$1
DB="nhl_hist_20072008-20192020.precovid.scored.db"
# evaluation against 20182019
SEASONS="20172018 20162017 20152016 20142015 20132014 20122013"
OUT_STAT=$2

if [ "$OUT_STAT" != "win" -a "$OUT_STAT" != "goal" ]; then
    echo "Unrecognized output stat '$OUT_STAT'"
    exit 1
fi


# MAKE SURE THIS IS ACCURATE OR HIGHER, total cases = 13598
MAX_OLS_FEATURES=48
MAX_CASES=10000
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")
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
team_home_H
"

CMD="$CMD
-o nhl_team_${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat $OUT_STAT
"

echo $CMD
