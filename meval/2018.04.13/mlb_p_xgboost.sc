for OUTSTAT in dk_score# fd_score#; do

    meval_O.sc --progress --cache_dir "./casedata_cache" \
               --scoring mae r2 --seasons 2017 2016 \
               --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
               --search_bayes_scorer mae \
               --folds 3 -o mlb_fantasy_p_xgboost \
               mlb.db xgboost \
               --model_player_stat $OUTSTAT \
               --player_pos P \
               --in_player_stats p_ip p_qs p_win p_er p_k p_hbp p_bb p_hits p_hr p_strikes p_wp \
               --in_team_stats p_win p_save p_hold errors \
               --cur_opp_team_stats off_1b off_2b off_3b off_hr off_rbi off_runs off_bb off_sb \
               off_k off_rbi_w2 off_rlob off_sac \
               --extra_stats starter_phand \
               --n_cases_range 500 5000 \
               --n_games_range 1 6 \
               --learning_rate_range .01 .2 \
               --subsample_range .5 1 \
               --min_child_weight_range 1 10 \
               --max_depth_range 3 10 \
               --gamma_range 0 10000 --gamma_range_def 10 log \
               --colsample_bytree_range 0.5 1 \
               --rounds_range 75 150

done
