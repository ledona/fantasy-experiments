# RUN ON WINDOWS

CACHE_DIR=~/windows_home/working/fantasy/casedata_cache
DB=~/windows_home/working/fantasy/mlb.db

for OUTSTAT in fd_score# dk_score#; do

    python -O ".\scripts\meval_O.sc" --progress --cache_dir "./casedata_cache" \
           --scoring mae --seasons 2016 2015 2014 2013 2012 \
           --search_method bayes --search_iters 50 --search_bayes_init_pts 5 \
           --search_bayes_scorer mae \
           --folds 3 -o nba_fantasy_dnn_sgd \
           nba.db keras \
           --model_player_stat $OUTSTAT \
           --n_games_range 1 6 \
           --n_cases_range 500 10000 --n_cases_range_inc 500 \
           --in_player_stats asst blks home "*_reb" "fg*" fouls "ft_*" mins pts starter \
           stls "tfg*" turnovers \
           --in_team_stats "*" --cur_opp_team_stats "*" \
           --hist_agg none \
           --normalize \
           --steps_range 100 1000 --steps_range_inc 100 \
           --layers_range 1 5 \
           --units_range 20 100 \
           --activation_list linear relu tanh sigmoid \
           --dropout_range .3 .7 \
           --learning_method_list sgd \
           --lr_range .0005 .001

done
