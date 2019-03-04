for OUTSTAT in dk_score# fd_score#; do

    meval_O.sc --progress --cache_dir "./casedata_cache" \
               --scoring mae r2 --seasons 2017 2016 \
               --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
               --search_bayes_scorer mae \
               --folds 3 -o mlb_fantasy_off_ble \
               mlb.db sklearn \
               --model_player_stat $OUTSTAT \
               --player_pos LF CF RF 1B 2B 3B SS C \
               --player_stats off_1b off_2b off_3b off_hr off_rbi off_runs off_hbp \
               off_bb off_sb off_k off_rbi_w2 off_rlob off_sac off_sb_c \
               --cur_opp_team_stats p_ip p_hits p_er p_k p_bb p_hr p_pc p_strikes \
               p_wp p_hbp p_win p_qs errors \
               --extra_stats opp_starter_p_ip opp_starter_p_hits opp_starter_p_er opp_starter_p_k \
               opp_starter_p_bb opp_starter_p_hr opp_starter_p_pc opp_starter_p_strikes \
               opp_starter_p_wp opp_starter_p_hbp opp_starter_p_win opp_starter_p_qs opp_starter_phand \
               --n_cases_range 500 20000 \
               --n_games_range 1 6 \
               --hist_agg_list mean median none \
               --alpha_range .00001 1  --alpha_range_def 6 log \
               --l1_ratio_range .05 .95 \
               --est_list br lasso elasticnet
done