for OUTSTAT in dk_score# fd_score#; do

    meval_O.sc --progress --cache_dir "./casedata_cache" \
               --scoring mae r2 --seasons 2016 2015 2014 2013 2012 \
               --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
               --search_bayes_scorer mae \
               --folds 3 -o nba_fantasy_ble \
               nba.db sklearn \
               --model_player_stat $OUTSTAT \
               --n_games_range 1 6 \
               --n_cases_range 500 10000 --n_cases_range_inc 500 \
               --in_player_stats asst blks home "*_reb" "fg*" fouls "ft_*" mins pts starter \
               stls "tfg*" turnovers \
               --in_team_stats "*" --cur_opp_team_stats "*" \
               --hist_agg_list none mean median \
               --alpha_range .00001 1  --alpha_range_def 6 log \
               --l1_ratio_range .05 .95 \
               --est_list br lasso elasticnet

done
