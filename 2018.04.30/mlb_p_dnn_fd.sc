OUTSTAT=fd_score#

meval_O.sc --progress --cache_dir "./casedata_cache" \
           --scoring mae --seasons 2017 2016 \
           --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
           --search_bayes_scorer mae \
           --folds 3 -o mlb_fantasy_p_dnn_rs \
           mlb.db keras \
           --model_player_stat $OUTSTAT \
           --player_pos P \
           --player_stats p_ip p_qs p_win p_er p_k p_hbp p_bb p_hits p_hr p_strikes p_wp \
           --team_stats p_win p_save p_hold errors \
           --cur_opp_team_stats off_1b off_2b off_3b off_hr off_rbi off_runs off_bb off_sb \
           off_k off_rbi_w2 off_rlob off_sac \
           --extra_stats starter_phand \
           --n_cases_range 500 5000 \
           --n_games_range 1 6 \
           --hist_agg none \
           --normalize \
           --steps_range 100 1000 --steps_range_inc 100 \
           --layers_range 1 5 \
           --units_range 20 100 \
           --activation_list linear relu tanh sigmoid \
           --dropout_range .3 .7 \
           --learning_method_list rmsprop sgd \
           --lr_range .0005 .01


meval_O.sc --progress --cache_dir "./casedata_cache" \
           --scoring mae --seasons 2017 2016 \
           --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
           --search_bayes_scorer mae \
           --folds 3 -o mlb_fantasy_p_dnn_ada \
           mlb.db keras \
           --model_player_stat $OUTSTAT \
           --player_pos P \
           --player_stats p_ip p_qs p_win p_er p_k p_hbp p_bb p_hits p_hr p_strikes p_wp \
           --team_stats p_win p_save p_hold errors \
           --cur_opp_team_stats off_1b off_2b off_3b off_hr off_rbi off_runs off_bb off_sb \
           off_k off_rbi_w2 off_rlob off_sac \
           --extra_stats starter_phand \
           --n_cases_range 500 5000 \
           --n_games_range 1 6 \
           --hist_agg none \
           --normalize \
           --steps_range 100 1000 --steps_range_inc 100 \
           --layers_range 1 5 \
           --units_range 20 100 \
           --activation_list linear relu tanh sigmoid \
           --dropout_range .3 .7 \
           --learning_method_list adagrad adadelta adam adamax nadam
