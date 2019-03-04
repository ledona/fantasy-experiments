#!/bin/bash

# disable expansion so SHARED_ARGS wildcards are left untouched
set -f

# for this experiment will be used djshadow as the remote data repository
if [[ $HOSTNAME == djshadow* ]]; then
    REMOTE_CACHE=""
else
    REMOTE_CACHE="--cache_remote djshadow:working/fantasy/casedata_cache"
fi

SHARED_ARGS="--progress --cache_dir ./casedata_cache $REMOTE_CACHE
           --scoring mae r2
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --seasons 20142015 20152016 20162017 20172018
           --folds 3"
SHARED_EXTRA_STATS="team_win home_C player_home_H"

# skaters
TYPE_S="--player_pos LW RW C D
        --player_stats goal goal_sh goal_w goal_t
                       assist* pen* toi_ev toi_pp toi_sh *away fo* hit pm shot*
        --team_stats ot so goal goal_pp pp pen* shot *away fo*
        --cur_opp_team_stats ot so goal goal_ag fo* goal_pk_ag pk pen*
                             hit shot_ag shot_b save *away"
OLS_FEATURES_S=51
MAX_CASES_S=55750
EXTRAS_S=""


# goalies
TYPE_G="--player_pos G
        --player_stats toi_g loss goal_ag save
        --team_stats ot so goal *_ag goal_pp fo* pp pk pen* shot_b *away fo* pen* hit
        --cur_opp_team_stats ot so goal fo* *pp goal_sh shot pen* hit *away"
OLS_FEATURES_G=37
MAX_CASES_G=3340
EXTRAS_G="player_win"

CALC_OLS='sklearn --est ols
          --hist_agg_list mean median'

CALC_BLE='sklearn
          --hist_agg_list mean median none
          --alpha_range .00001 1  --alpha_range_def 6 log
          --l1_ratio_range .05 .95
          --est_list br lasso elasticnet'

CALC_RF='sklearn
         --extra_stats venue_C
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
    echo "Create the cmd line meval to run. First arg is for 'G'oalie or 'S'kater
usage: nba.sc (G|S) (OLS|RF|XG|BLE|DNN_RS|DNN_ADA) (dk|fd|y)"
}

TYPE=TYPE_${1}
CALC=CALC_${2}

if [ -z "${!TYPE}" ] || [ -z "${!CALC}" ] || [ "$3" != "dk" -a "$3" != "fd" -a "$3" != "y" ]; then
    usage
    exit 1
fi

MAX_CASES=MAX_CASES_${1}
SHARED_CALC="--n_games_range 1 7 --n_cases_range 500 ${!MAX_CASES}"

OLS_MAX_FEATURES=OLS_FEATURES_${1}
EXTRAS="EXTRAS_${1}"

EXTRA_STATS="${!EXTRAS}"
if [ "$2" != "OLS" ] && [ "$1" == "S" ]; then
    EXTRA_STATS="${EXTRA_STATS} player_pos_C"
fi

if [ "$2" == "OLS" ]; then
    OLS_FEATURES=OLS_FEATURES_${1}
    FEATURES_ARG="--n_features_range 1 ${!OLS_FEATURES}"
else
    FEATURES_ARG=""
fi

if [ ! -z "${EXTRA_STATS}" ]; then
    EXTRA_STATS="--extra_stats ${EXTRA_STATS}"
fi

CMD="python -O scripts/meval.sc $SHARED_ARGS -o nhl_${1}_${2} nhl.db ${!CALC} ${!TYPE}
$SHARED_CALC $EXTRA_STATS --model_player_stat ${3}_score# $FEATURES_ARG"

echo $CMD
