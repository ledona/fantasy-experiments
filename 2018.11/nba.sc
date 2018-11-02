#!/bin/bash

# disable expansion so SHARED_ARGS wildcards are left untouched
set -f

REMOTE_CACHE=""

# for this experiment will be used djshadow as the remote data repository
# if [[ $HOSTNAME == djshadow* ]]; then
#     REMOTE_CACHE=""
# else
#     REMOTE_CACHE="--cache_remote djshadow:working/fantasy/casedata_cache"
# fi

SHARED_ARGS="--progress --cache_dir ./casedata_cache $REMOTE_CACHE
           --scoring mae r2
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --seasons 20142015 20152016 20162017 20172018
           --folds 3"

SHARED_CALC="--n_games_range 1 7
        --player_stats *
        --team_stats *
        --cur_opp_team_stats *
        --n_cases_range 500 32000"

CALC_OLS='sklearn --est ols
        --n_features_range 1 65
        --hist_agg_list mean median'

CALC_BLE='sklearn
        --hist_agg_list mean median none
        --alpha_range .00001 1  --alpha_range_def 6 log
        --l1_ratio_range .05 .95
        --est_list br lasso elasticnet'

CALC_RF='sklearn
       --hist_agg_list mean median none
       --rf_trees_range 5 25 --rf_max_features_list sqrt log2
       --rf_min_samples_leaf_range 1 200
       --rf_crit_list mse mae --rf_max_depth_list 0 500
       --rf_n_jobs 3
       --est rforest'

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

CALC_DNN_ADA="$_SHARED_DNN
            --learning_method_list adagrad adadelta adam adamax nadam"

CALC_XG="xgboost
       --hist_agg_list mean median none
       --learning_rate_range .01 .2
       --subsample_range .5 1
       --min_child_weight_range 1 10
       --max_depth_range 3 10
       --gamma_range 0 10000 --gamma_range_def 10 log
       --colsample_bytree_range 0.5 1
       --rounds_range 75 150"


usage()
{
    echo "Create the cmd line meval to run.
usage: nba.sc (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y)"
}

CALC=CALC_${1}

if [ -z "${!CALC}" ] || [ "$2" != "dk" -a "$2" != "fd" -a "$2" != "y" ]; then
    usage
    exit 1
fi

EXTRA_STATS="--extra_stats team_win home_C player_home_H"

CMD="python -O scripts/meval.sc $SHARED_ARGS -o nba_${1} nba.db ${!CALC}
$SHARED_CALC $EXTRA_STATS --model_player_stat ${2}_score#"

echo $CMD
