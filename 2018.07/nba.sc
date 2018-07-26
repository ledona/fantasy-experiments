#!/bin/bash

SHARED_ARGS='--progress --cache_dir "./casedata_cache"  --scoring mae r2
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --seasons 2015 2016 2017 2018
           --folds 3'

SHARED_CALC="--n_games_range 1 7
        --player_stats off_*
        --team_stats off_runs off_hit off_bb
        --cur_opp_team_stats p_* errors
        --extra_stats $EXTRA_STATS
        --n_cases_range 500 40000"
EXTRAS_OFF="off_hit_side opp_starter_*"
SEASONS_OFF="--seasons 2017 2016 2015"

TYPE_P="$N_GAMES --player_pos P
      --player_stats p_*
      --team_stats p_win p_runs p_save errors
      --cur_opp_team_stats off_*
      --n_cases_range 500 10000"
EXTRAS_P="starter_phand_C opp_*_hit_%_* player_win"
SEASONS_P="--seasons 2017 2016 2015 2014"

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
usage: mlb.sc (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y)"
}

CALC=CALC_${1}

if [ -z "${!CALC}" ] || [ "$3" != "dk" -a "$3" != "fd" -a "$3" != "y" ]; then
    usage
    exit 1
fi

EXTRA_STATS_TYPE_NAME=EXTRAS_${1}
EXTRA_STATS_CALC_NAME=EXTRAS_${2}
EXTRA_STATS="$SHARED_EXTRAS ${!EXTRA_STATS_TYPE_NAME} ${!EXTRA_STATS_CALC_NAME}"

CMD="python -O scripts/meval.sc $SHARED_ARGS ${!SEASONS} -o nba_${1}_${2} mlb.db ${!CALC} ${!TYPE}
--model_player_stat ${3}_score#"

echo $CMD
