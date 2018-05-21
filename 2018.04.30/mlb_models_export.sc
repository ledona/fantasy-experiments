# calc def id=3 name='DKOff'
train.sc --new_calc_def_name 'DKOff' --seasons 2017 2016 --season_parts REG -- mlb.db sklearn --player_pos 1B 2B 3B C CF LF RF SS --model_player_stat dk_score#1 --player_stats off_1b off_2b off_3b off_bb off_hbp off_hr off_k off_rbi off_rbi_w2 off_rlob off_runs off_sac off_sb off_sb_c --cur_opp_team_stats errors p_bb p_er p_hbp p_hits p_hr p_ip p_k p_pc p_qs p_strikes p_win p_wp --extra_stats opp_starter_p_bb opp_starter_p_er opp_starter_p_hbp opp_starter_p_hits opp_starter_p_hr opp_starter_p_ip opp_starter_p_k opp_starter_p_pc opp_starter_p_qs opp_starter_p_strikes opp_starter_p_win opp_starter_p_wp opp_starter_phand --hist_agg median --n_games 6 --n_cases 500 --est ols --alpha 1.0 --l1_ratio 0.5 --rf_trees 10 --rf_max_features sqrt --rf_min_samples_leaf 1 --rf_crit mse --rf_max_depth 0 --rf_n_jobs 2

# calc def id=4 name='DKP'
train.sc --new_calc_def_name 'DKP' --seasons 2017 2016 --season_parts REG -- mlb.db sklearn --player_pos P --model_player_stat dk_score#1 --player_stats p_bb p_er p_hbp p_hits p_hr p_ip p_k p_qs p_strikes p_win p_wp --cur_opp_team_stats off_1b off_2b off_3b off_bb off_hr off_k off_rbi off_rbi_w2 off_rlob off_runs off_sac off_sb --extra_stats starter_phand --hist_agg mean --n_games 4 --n_cases 500 --est br --alpha 1.0 --l1_ratio 0.3213809029 --rf_trees 10 --rf_max_features sqrt --rf_min_samples_leaf 1 --rf_crit mse --rf_max_depth 0 --rf_n_jobs 2

# calc def id=5 name='FDP'
train.sc --new_calc_def_name 'FDP' --seasons 2017 2016 --season_parts REG -- mlb.db sklearn --player_pos P --model_player_stat fd_score#2 --player_stats p_bb p_er p_hbp p_hits p_hr p_ip p_k p_qs p_strikes p_win p_wp --cur_opp_team_stats off_1b off_2b off_3b off_bb off_hr off_k off_rbi off_rbi_w2 off_rlob off_runs off_sac off_sb --extra_stats starter_phand --hist_agg mean --n_games 5 --n_cases 500 --est lasso --alpha 1.0 --l1_ratio 0.95 --rf_trees 10 --rf_max_features sqrt --rf_min_samples_leaf 1 --rf_crit mse --rf_max_depth 0 --rf_n_jobs 2

# calc def id=7 name='FDOff'
train.sc --new_calc_def_name 'FDOff' --seasons 2017 --season_parts REG -- mlb.db keras --player_pos 1B 2B 3B C CF LF RF SS --model_player_stat fd_score#2 --player_stats off_1b:1 off_2b:1 off_3b:1 off_bb:1 off_hbp:1 off_hr:1 off_k:1 off_rbi:1 off_rbi_w2:1 off_rlob:1 off_runs:1 off_sac:1 off_sb:1 off_sb_c:1 --cur_opp_team_stats errors:1 p_bb:1 p_er:1 p_hbp:1 p_hits:1 p_hr:1 p_ip:1 p_k:1 p_pc:1 p_qs:1 p_strikes:1 p_win:1 p_wp:1 --extra_stats opp_starter_p_bb:1 opp_starter_p_er:1 opp_starter_p_hbp:1 opp_starter_p_hits:1 opp_starter_p_hr:1 opp_starter_p_ip:1 opp_starter_p_k:1 opp_starter_p_pc:1 opp_starter_p_qs:1 opp_starter_p_strikes:1 opp_starter_p_win:1 opp_starter_p_wp:1 opp_starter_phand:1 --hist_agg none --n_games 1 --n_cases 7881 --steps 200 --units 100 --layers 3 --dropout 0.4450377674 --activation relu --learning_method adamax --models_path /Users/delano/working/fantasy/MODELS_keras
