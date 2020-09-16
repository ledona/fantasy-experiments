#!/bin/bash

script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "LOL Player meval
usage: $(basename "$0") ($AVAILABLE_MODELS) (ALL|TOP|MID|JNG|SUP|ADCB) (dk|fd) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 2 ]; then
    usage
    echo Missing required args
    exit 1
fi

# set environment variables needed for analysis
SEASONS="2014 2015 2016 2017 2018 2019"
DB="lol_hist_2014-2019.scored.db"
MODEL=$1
POS=$2
SERVICE=$3


# total cases for all = 108272; for a position = 21373
if [ "$MODEL" == "GP" ]; then
    MAX_CASES=15000
else
    if [ "$POS" == "ALL" ]; then
        MAX_CASES=100000
    else
        MAX_CASES=20000
    fi
fi
MAX_OLS_FEATURES=74
source ${script_dir}/env.sc

if [ "$?" -eq 1 ] ||
   [ "$SERVICE" != "dk" -a "$SERVICE" != "fd" ]; then
    usage
    echo Error parsing args or getting shared settings
    exit 1
fi

CALC_ARGS=$(get_calc_args "$MODEL" "$4") && CMD=$(get_meval_base_cmd "$4")

PLAYER_STATS="asst
cs csat10 csat15 csdat10 csdat15 cspm
d dmgpm dsh
gat10 gat15 gdat10 gdat15 gepm
k ka10+
mbo mgp w
xpat10 xpat15"

TEAM_STATS="brnk
csat10 csat15 csdat10 csdat15 cspm
d dmgpm dpm drgk
gat10 gat15 gdat10 gdat15 gepm
k kpm
tk
vspm
wdcpm wdpm
xpat10 xpat15 xpdat10 xpdat15"

EXTRA_STATS="
modeled_stat_std_mean
modeled_stat_trend
"

CUR_OPP_TEAM_STATS=$TEAM_STATS

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS player_pos_C"
fi


CMD="$CMD $DATA_FILTER_FLAG
-o lol-player-${POS}-${MODEL}
${DB}
${CALC_ARGS}
--player_stats $PLAYER_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--team_stats $TEAM_STATS
--extra_stats $EXTRA_STATS
--model_player_stat ${SERVICE}_performance_score#
"

if [ "$POS" != "ALL" ]; then
    CMD="$CMD --player_pos $POS"
fi

echo $CMD
