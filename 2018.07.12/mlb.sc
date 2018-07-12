#!/bin/bash

SHARED_ARGS='--progress --cache_dir "./casedata_cache"  --scoring mae r2 --seasons 2017 2016 2015
           --search_method bayes --search_iters 70 --search_bayes_init_pts 7
           --search_bayes_scorer mae
           --folds 3'

TYPE_OFF='--player_pos LF CF RF 1B 2B 3B SS C
        --player_stats off_1b off_2b off_3b off_hr off_rbi off_runs off_hbp
        off_bb off_sb off_k off_rbi_w2 off_rlob off_sac off_sb_c
        --cur_opp_team_stats p_ip p_hits p_er p_k p_bb p_hr p_pc p_strikes
        p_wp p_hbp p_win p_qs errors
        --extra_stats off_hit_side opp_starter_p_ip opp_starter_p_hits opp_starter_p_er opp_starter_p_k
        opp_starter_p_bb opp_starter_p_hr opp_starter_p_pc opp_starter_p_strikes
        opp_starter_p_wp opp_starter_p_hbp opp_starter_p_win opp_starter_p_loss opp_starter_p_qs
        opp_starter_phand_C opp_starter_phand_H *_home_* team_win
        --n_cases_range 500 40000
        --n_games_range 1 7'

TYPE_P='--player_pos P
      --player_stats p_ip p_qs p_win p_loss p_er p_k p_hbp p_bb p_hits p_hr p_strikes p_wp
      --team_stats p_win p_save p_hold errors
      --cur_opp_team_stats off_1b off_2b off_3b off_hr off_rbi off_runs off_bb off_sb
      off_k off_rbi_w2 off_rlob off_sac
      --extra_stats starter_phand_C opp_*_hit_%_* *_home_* team_win
      --n_cases_range 500 10000
      --n_games_range 1 7'

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
           --extra_stats venue_H venue_C'

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

if [ -z "${!TYPE}" ] || [ -z "${!CALC}" ] || [ "$3" != "dk" -a "$3" != "fd" ]; then
    usage
    exit 1
fi

CMD="python -O scripts/meval.sc $SHARED_ARGS -o mlb_${1}_${2} mlb.db ${!CALC} ${!TYPE} 
--model_player_stat ${3}_score#"

echo $CMD
