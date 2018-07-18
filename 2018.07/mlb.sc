#!/bin/bash

SHARED_ARGS='--progress --cache_dir "./casedata_cache"  --scoring mae r2
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --folds 3'

SHARED_TYPE_ARGS="--n_games_range 1 7 --extra_stats *home* team_win venue_C"

# Input stats for offensive players tries to account for team offense production, player's
# offense, opposing pitcher, opposing team and where the games are happening
TYPE_OFF="$SHARED_TYPE_ARGS --player_pos LF CF RF 1B 2B 3B SS C
        --player_stats off_*
        --team_stats off_runs off_hit off_bb
        --cur_opp_team_stats p_* errors
        --extra_stats off_hit_side opp_starter_*
        --n_cases_range 500 40000"

SEASONS_OFF="--seasons 2017 2016 2015"

TYPE_P="$SHARED_TYPE_ARGS --player_pos P
      --player_stats p_*
      --team_stats p_win p_runs p_save errors
      --cur_opp_team_stats off_*
      --extra_stats starter_phand_C opp_*_hit_%_* player_win
      --n_cases_range 500 10000"

SEASONS_P="--seasons 2017 2016 2015 2014"

CALC_OLS='sklearn --est ols
        --n_features_range 1 45
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
           --dropout_range .3 .7
           --extra_stats venue_H'

CALC_DNN_RS="$_SHARED_DNN
           --learning_method_list rmsprop sgd
           --lr_range .0005 .01"

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
usage: mlb.sc (OFF|P) (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd)"
}

TYPE=TYPE_${1}
CALC=CALC_${2}
SEASONS=SEASONS_${1}

if [ -z "${!TYPE}" ] || [ -z "${!CALC}" ] || [ "$3" != "dk" -a "$3" != "fd" ]; then
    usage
    exit 1
fi

CMD="python -O scripts/meval.sc $SHARED_ARGS ${!SEASONS} -o mlb_${1}_${2} mlb.db ${!CALC} ${!TYPE}
--model_player_stat ${3}_score#"

echo $CMD
