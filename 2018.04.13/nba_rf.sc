OUTSTAT=dk_score#

meval_O.sc  --progress --cache_dir "./casedata_cache" \
--scoring mae r2 --seasons 2016 2015 2014 2013 2012 \
--search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
--search_bayes_scorer mae \
--folds 3 -o nba_fantasy_rf \
nba.db sklearn \
--model_player_stat $OUTSTAT \
--n_games_range 1 6 \
--n_cases_range 500 10000 --n_cases_range_inc 500 \
--in_player_stats asst blks home "*_reb" "fg*" fouls "ft_*" mins pts starter \
   stls "tfg*" turnovers \
--in_team_stats "*" --cur_opp_team_stats "*" \
--rf_trees_range 5 25 --rf_max_features_list sqrt log2 \
--rf_min_samples_leaf_range 1 200 \
--rf_crit_list mse mae --rf_max_depth_list 0 500 \
--rf_n_jobs 3 \
--est rforest
