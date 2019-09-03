#!/bin/bash

SHARED_ARGS="--progress --cache_dir ./casedata_cache --scoring mae r2
           --slack
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --folds 3"

N_GAMES="--n_games_range 1 7"
SHARED_EXTRAS="home_C team_home_H"

# Input stats for offensive players tries to account for team offense production, player's
# offense, opposing pitcher, opposing team and where the games are happening
TYPE="$N_GAMES
      --n_cases_range 500 49000
      --team_stats off_1b off_2b off_3b off_bb off_hit off_hr off_k off_r* off_sac* win
      --cur_opp_team_stats errors p_* win"

EXTRAS="opp_starter_*"
SEASONS="--seasons 2018 2017 2016 2015 2014"

CALC_OLS='sklearn --est ols
        --hist_agg_list mean median'
EXTRAS_OLS=""

CALC_BLE='sklearn
        --hist_agg_list mean median none
        --alpha_range .00001 1  --alpha_range_def 6 log
        --l1_ratio_range .05 .95
        --est_list br lasso elasticnet'
EXTRAS_BLE=""

CALC_RF='sklearn
       --hist_agg_list mean median none
       --rf_trees_range 5 25 --rf_max_features_list sqrt log2
       --rf_min_samples_leaf_range 1 200
       --rf_crit_list mse mae --rf_max_depth_list 0 500
       --rf_n_jobs 3
       --est rforest'
EXTRAS_RF=""

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
EXTRAS_DNN_RS="venue_H"

CALC_DNN_ADA="$_SHARED_DNN
            --learning_method_list adagrad adadelta adam adamax nadam"
EXTRAS_DNN_ADA="venue_H"

CALC_XG="xgboost
       --hist_agg_list mean median none
       --learning_rate_range .01 .2
       --subsample_range .5 1
       --min_child_weight_range 1 10
       --max_depth_range 3 10
       --gamma_range 0 10000 --gamma_range_def 10 log
       --colsample_bytree_range 0.5 1
       --rounds_range 75 150"
EXTRAS_XG=""


usage()
{
    echo "Create the cmd line meval to run.
usage: mlb.sc (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y)"
}

CALC=CALC_${1}

if [ -z "${!TYPE}" ] || [ -z "${!CALC}" ] || [ "$2" != "dk" -a "$2" != "fd" -a "$2" != "y" ]; then
    usage
    exit 1
fi

EXTRA_STATS_CALC_NAME=EXTRAS_${1}
EXTRA_STATS="$SHARED_EXTRAS ${!EXTRA_STATS_TYPE_NAME} ${!EXTRA_STATS_CALC_NAME}"

if [ "$1" != "OLS" ]; then
    FEATURES_ARG=""
    # add current venue to every offensive thing except OLS which doesn't support categoricals
    EXTRA_STATS="$EXTRA_STATS venue_C"
else
    FEATURES_ARG="--n_features_range 1 77"
fi

CMD="python -O scripts/meval.sc $SHARED_ARGS ${!SEASONS} -o mlb_${1}_${2}
mlb_hist_2008-2018.db ${!CALC} ${!TYPE} --model_player_stat ${2}_score#
--extra_stats $EXTRA_STATS $FEATURES_ARG"

echo $CMD
