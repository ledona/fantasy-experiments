for OUTSTAT in dk_score# fd_score#; do

    meval_O.sc --progress --cache_dir "./casedata_cache" \
               --scoring mae r2 --seasons 2016 2015 2014 2013 2012 \
               --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
               --search_bayes_scorer mae \
               --folds 3 -o nba_fantasy_ols \
               nba.db sklearn \
               --model_player_stat $OUTSTAT \
               --n_games_range 1 6 \
               --n_cases_range 500 10000 --n_cases_range_inc 500 \
               --player_stats asst blks home "*_reb" "fg*" fouls "ft_*" mins pts starter \
               stls "tfg*" turnovers \
               --team_stats "*" --cur_opp_team_stats "*" \
               --n_features_range 1 42 \
               --hist_agg_list mean median \
               --est ols

done
