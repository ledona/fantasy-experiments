#!/bin/bash

# Use this script to run individual nfl evaluations

SHARED_MEVAL_ARGS='--progress --cache_dir ./casedata_cache  --scoring mae r2
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --folds 2 --seasons 2017 2016 2015 2014 2013 2012 2011 2010 2009'

TYPE_QB="--player_pos QB
       --player_stats fumbles_lost passing_* rushing_* tds
       --team_stats pts rushing_yds turnovers
       --cur_opp_team_stats def_* op_* yds pts turnovers pens pen_yds"
MAX_CASES_QB=1750
OLS_FEATURES_QB=39

TYPE_WT="--player_pos WR TE
       --player_stats fumbles_lost receiving_* tds
       --team_stats passing_yds pts rushing_yds turnovers
       --cur_opp_team_stats def_* op_* pens pen_yds"
MAX_CASES_WT=9000
OLS_FEATURES_WT=32

TYPE_RB="--player_pos RB
       --player_stats fumbles_lost receiving_* rushing_* tds
       --team_stats passing_yds pts rushing_yds turnovers
       --cur_opp_team_stats def_* op_* pens pen_yds"
MAX_CASES_RB=3500
OLS_FEATURES_RB=36

TYPE_D="--team_stats def_* op_* pts yds turnovers pens pen_yds
      --cur_opp_team_stats passing_yds pts rushing_yds turnovers pens pen_yds"
MAX_CASES_D=1750
OLS_FEATURES_D=30

CALC_OLS='sklearn --est ols
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
usage: nfl.sc (QB|WT|R|D) (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y)"
}

TYPE=TYPE_${1}
CALC=CALC_${2}

if [ -z "${!TYPE}" ] || [ -z "${!CALC}" ] || [ "$3" != "dk" -a "$3" != "fd" -a "$3" != "y" ]; then
    usage
    exit 1
fi

if [ "$1" == "D" ]; then
    MODEL_ARG="--model_team_stat ${3}_score_def#"
else
    MODEL_ARG="--model_player_stat ${3}_score_off#"
fi

if [ "$2" == "OLS" ]; then
    OLS_FEATURES=OLS_FEATURES_${1}
    FEATURES_ARG="--n_features_range 1 ${!OLS_FEATURES}"
else
    FEATURES_ARGS=""
fi


MAX_CASES=MAX_CASES_${1}
SHARED_CALC_ARGS="--n_games_range 1 7 --n_cases_range 100 ${!MAX_CASES} --extra_stats *home* team_win"


CMD="python -O scripts/meval.sc $SHARED_MEVAL_ARGS -o nfl_${1}_${2} nfl.db
     ${!CALC} $SHARED_CALC_ARGS ${!TYPE} $MODEL_ARG $FEATURES_ARG"

echo $CMD
