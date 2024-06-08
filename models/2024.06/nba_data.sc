#!/bin/bash

DB_FILE=${FANTASY_HOME}/nba_hist_20082009-20222023.scored.db
SEASONS="20152016 20162017 20172018 20182019 20192020 20202021 20212022 20222023"

# player
dumpdata.sc --seasons $SEASONS --progress $DB_FILE --no_teams \
    --current_extra is_home "game_pos_*_rank" \
    --stats "*" --target_calc_stats "*" --target_stats pts asst d_reb o_reb fg_att \
    --player_team_stats "*" --opp_team_stats "*" \
    --hist_recent_games 5 --hist_recent_mode ma \
    --dask_inf_multi_season_mode processes \
    --dask_tasks 4 \
    --format parquet -f nba_player.pq

# team
dumpdata.sc --seasons $SEASONS --progress $DB_FILE --no_players \
    --stats "*" --target_stats pts win \
    --opp_team_stats "*" \
    --hist_recent_games 5 --hist_recent_mode ma --current_extra is_home \
    --format parquet -f nba_team.pq