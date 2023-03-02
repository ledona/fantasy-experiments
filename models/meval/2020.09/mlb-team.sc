#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/const.sc

usage()
{
    echo "MLB team score prediction meval.
usage: $(basename "$0") ($AVAILABLE_MODELS) (win|runs) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

if [ "$#" -lt 2 ]; then
    usage
    exit 1
fi

MODEL=$1
DB="mlb_hist_2008-2019.scored.db"
# evaluations against 2018
SEASONS="2019 2017 2016 2015 2014"
OUT_STAT=$2
if [ $OUT_STAT == 'runs' ]; then
    OUT_STAT="off_runs"
elif [ $OUT_STAT != 'win' ]; then
    echo "Output stat '$OUT_STAT' is not valid!"
    exit 1
fi

# MAKE SURE THIS IS ACCURATE OR HIGHER
MAX_OLS_FEATURES=65
# total cases 24076
MAX_CASES=16000
source ${script_dir}/env.sc

# parse the command line
CALC_ARGS=$(get_calc_args "$MODEL" "$3") && CMD=$(get_meval_base_cmd "$3")
ERROR=$?

if [ "$ERROR" -eq 1 ]; then
    usage
    exit 1
fi


TEAM_STATS="win off_1b off_2b off_3b off_ab off_bb off_hbp
            off_hit off_hr off_k off_pa off_rbi
            off_rbi_w2 off_rlob off_runs off_sac off_sac_f
            off_sac_h off_sb off_sb_c"
CUR_OPP_TEAM_STATS="errors p_bb p_cg p_er p_hbp p_hits p_hold p_hr p_ibb p_ip p_k
                    p_loss p_pc p_qs p_runs p_save p_strikes"
EXTRA_STATS="modeled_stat_trend modeled_stat_std_mean
             home_C l_hit_%_C l_hit_%_H
             opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
             opp_starter_p_bb opp_starter_p_cg opp_starter_p_er opp_starter_p_hbp
             opp_starter_p_hits opp_starter_p_hr opp_starter_p_ibb opp_starter_p_ip
             opp_starter_p_k opp_starter_p_loss opp_starter_p_pc opp_starter_p_qs
             opp_starter_p_runs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp
             r_hit_%_C r_hit_%_H
             team_home_H"

if [ "$MODEL" != "OLS" ]; then
    # include categorical features, not supported for OLS due to lack of feature selection support
    EXTRA_STATS="$EXTRA_STATS venue_C
                 opp_starter_phand_C opp_starter_phand_H
                 starter_phand_C"
fi

CMD="$CMD
-o mlb-team-${1}
${DB}
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat $OUT_STAT
"

echo $CMD
