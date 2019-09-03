#!/bin/bash

# set environment variables needed for analysis
script_dir="$(dirname "$0")"
source ${script_dir}/mlb-env.sc

usage()
{
    script_name=
    echo "MLB team score prediction meval.
usage: $(basename "$0") (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) [--test]

--test - (optional) Do a short test (fewer seasons, iterations, etc)
"
}

TEST_ARG=$2

if [ "$TEST_ARG" != "--test" -a "$TEST_ARG" != "" ]; then
    usage
    exit 1
fi

CALC_ARGS=$(get_calc_args "$1" "$TEST_ARG")

CMD=$(get_meval_base_cmd "$TEST_ARG")


TEAM_STATS="win off_1b off_2b off_3b off_ab off_bb off_hbp
            off_hit off_hr off_k off_pa off_rbi
            off_rbi_w2 off_rlob off_runs off_sac off_sac_f
            off_sac_h off_sb off_sb_c"
CUR_OPP_TEAM_STATS="errors p_bb p_cg p_er p_hbp p_hits p_hold p_hr p_ibb p_ip p_k
                    p_loss p_pc p_qs p_runs p_save p_strikes"
EXTRA_STATS="home_C l_hit_%_C l_hit_%_H
             opp_l_hit_%_C opp_l_hit_%_H opp_r_hit_%_C opp_r_hit_%_H
             opp_starter_p_bb opp_starter_p_cg opp_starter_p_er opp_starter_p_hbp
             opp_starter_p_hits opp_starter_p_hr opp_starter_p_ibb opp_starter_p_ip
             opp_starter_p_k opp_starter_p_loss opp_starter_p_pc opp_starter_p_qs
             opp_starter_p_runs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp
             opp_starter_phand_C opp_starter_phand_H
             r_hit_%_C r_hit_%_H starter_phand_C
             team_home_H venue_H venue_C"

CMD="$CMD
-o mlb_team-score_${1}
mlb_hist_2008-2018.scored.db
${CALC_ARGS}
--team_stats $TEAM_STATS
--cur_opp_team_stats $CUR_OPP_TEAM_STATS
--extra_stats $EXTRA_STATS
--model_team_stat off_runs
"

echo $CMD
