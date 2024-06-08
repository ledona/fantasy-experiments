#!/bin/bash

DB_FILE=${FANTASY_HOME}/lol_hist_2014-2022.scored.db
SEASONS="2016 2017 2018 2019 2020 2021 2022"

# player data
dumpdata.sc --seasons $SEASONS --slack ${DB_FILE} --no_teams \
    --stats "*" --target_calc_stats "*performance_score" \
    --player_team_stats "*" --opp_team_stats "*" \
    --hist_recent_games 3 --hist_recent_mode ma \
    --dask_mode processes --dask_tasks 4 \
    --format parquet -f lol_player.pq

# team data
dumpdata.sc --seasons $SEASONS --slack ${DB_FILE} --no_players \
    --stats "*" --target_calc_stats "*performance_score" \
    --opp_team_stats "*" \
    --hist_extra match_win --current_extra match_win --hist_opp_extra match_win \
    --hist_recent_games 3 --hist_recent_mode ma \
    --dask_mode processes --dask_tasks 4 \
    --format parquet -f lol_team.pq