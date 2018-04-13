for OUTSTAT in dk_score# fd_score#; do
    meval_O.sc --progress --cache_dir "./casedata_cache" \
               --scoring mae --seasons 2016 2015 2014 2013 2012 \
               --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
               --search_bayes_scorer mae \
               --folds 3 -o nba_fantasy_xgboost \
               nba.db xgboost \
               --model_player_stat $OUTSTAT \
               --n_games_range 1 6 \
               --n_cases_range 500 10000 --n_cases_range_inc 500 \
               --in_player_stats asst blks home "*_reb" "fg*" fouls "ft_*" mins pts starter \
               stls "tfg*" turnovers \
               --in_team_stats "*" --cur_opp_team_stats "*" \
               --learning_rate_range .01 .2 \
               --subsample_range .5 1 \
               --min_child_weight_range 1 10 \
               --max_depth_range 3 10 \
               --gamma_range 0 10000 --gamma_range_def 10 log \
               --colsample_bytree_range 0.5 1 \
               --rounds_range 75 150
done
