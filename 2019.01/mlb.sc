#!/bin/bash

# for this experiment will be used djshadow as the remote data repository
if [[ $HOSTNAME == djshadow* ]]; then
    REMOTE_CACHE=""
else
    REMOTE_CACHE="--cache_remote djshadow:working/fantasy/casedata_cache"
fi

SHARED_ARGS="--progress --cache_dir ./casedata_cache $REMOTE_CACHE  --scoring mae r2
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --folds 3"

N_GAMES="--n_games_range 1 7"
SHARED_EXTRAS="*home*"

# Input stats for offensive players tries to account for team offense production, player's
# offense, opposing pitcher, opposing team and where the games are happening
TYPE_OFF="$N_GAMES --player_pos LF CF RF 1B 2B 3B SS C
        --player_stats off_*
        --team_stats off_1b off_2b off_3b off_bb off_hit off_hr off_k off_r* off_sac* win
        --cur_opp_team_stats errors p_* win
        --n_cases_range 500 49000"
EXTRAS_OFF="off_hit_side opp_starter_*"
SEASONS_OFF="--seasons 2018 2017 2016 2015"

TYPE_P="$N_GAMES --player_pos P
      --player_stats p_bb p_cg p_er p_hbp p_hits p_hr p_ibb p_ip p_k p_loss
                     p_pc p_po p_qs p_runs p_strikes p_win p_wp
      --team_stats errors off_runs p_cg p_er p_save p_hold win
      --cur_opp_team_stats off_* win
      --n_cases_range 500 31000"
EXTRAS_P="starter_phand_C opp_*_hit_%_* player_win"
SEASONS_P="--seasons 2018 2017 2016 2015 2014"

CALC_OLS='sklearn --est ols
        --n_features_range 1 45
        --hist_agg_list mean median'
EXTRAS_OLS=""

CALC_BLE='sklearn
        --hist_agg_list mean median none
        --alpha_range .00001 1  --alpha_range_def 6 log
        --l1_ratio_range .05 .95
        --est_list br lasso elasticnet'
EXTRAS_BLE="venue_C"

CALC_RF='sklearn
       --extra_stats venue_C
       --hist_agg_list mean median none
       --rf_trees_range 5 25 --rf_max_features_list sqrt log2
       --rf_min_samples_leaf_range 1 200
       --rf_crit_list mse mae --rf_max_depth_list 0 500
       --rf_n_jobs 3
       --est rforest'
EXTRAS_RF="venue_C"

_SHARED_DNN='keras
           --hist_agg none
           --normalize
           --steps_range 100 1000 --steps_range_inc 100
           --layers_range 1 5
           --units_range 20 100
           --activation_list linear relu tanh sigmoid
           --dropout_range .3 .7'

CALC_DNN_RS="$_SHARED_DNN
           --learning_method_list rmsprop sgd
           --lr_range .005 .01"
EXTRAS_DNN_RS="venue_C venue_H"

CALC_DNN_ADA="$_SHARED_DNN
            --learning_method_list adagrad adadelta adam adamax nadam"
EXTRAS_DNN_ADA="venue_C venue_H"

CALC_XG="xgboost
       --hist_agg_list mean median none
       --learning_rate_range .01 .2
       --subsample_range .5 1
       --min_child_weight_range 1 10
       --max_depth_range 3 10
       --gamma_range 0 10000 --gamma_range_def 10 log
       --colsample_bytree_range 0.5 1
       --rounds_range 75 150"
EXTRAS_XG="venue_C"


usage()
{
    echo "Create the cmd line meval to run.
usage: mlb.sc (OFF|P) (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y)"
}

TYPE=TYPE_${1}
CALC=CALC_${2}
SEASONS=SEASONS_${1}

if [ -z "${!TYPE}" ] || [ -z "${!CALC}" ] || [ "$3" != "dk" -a "$3" != "fd" -a "$3" != "y" ]; then
    usage
    exit 1
fi

EXTRA_STATS_TYPE_NAME=EXTRAS_${1}
EXTRA_STATS_CALC_NAME=EXTRAS_${2}
EXTRA_STATS="$SHARED_EXTRAS ${!EXTRA_STATS_TYPE_NAME} ${!EXTRA_STATS_CALC_NAME}"

if [ "$2" != "OLS" ]; then
    FEATURES_ARG=""
    if [ "$1" == "OFF" ]; then
        # add off hit side to every offensive thing except OLS which doesn't support categoricals
        EXTRA_STATS="$EXTRA_STATS off_hit_side venue*"
    fi
else
    if [ "$1" == "OFF" ]; then
        FEATURES_ARG="--n_features_range 1 81"
    else
        # should be pitchers
        FEATURES_ARG="--n_features_range 1 56"
    fi
fi

CMD="python -O scripts/meval.sc $SHARED_ARGS ${!SEASONS} -o mlb_${1}_${2}
mlb_modeling_2008-2018.db ${!CALC} ${!TYPE} --model_player_stat ${3}_score#
--extra_stats $EXTRA_STATS $FEATURES_ARG"

echo $CMD
