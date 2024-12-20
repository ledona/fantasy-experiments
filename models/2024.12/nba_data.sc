#!/bin/bash

DB_FILE=${FANTASY_HOME}/nba.hist.20082009-20232024.scored.db
DEST=/fantasy-isync/fantasy-modeling/2024.12/data
SEASONS="20082009 20092010 20102011 20112012 20122013 20132014 20142015 
20152016 20162017 20172018 20182019 20192020 20202021 20212022 20222023 20232024"

# player
dumpdata.sc $DB_FILE --seasons $SEASONS --no_teams \
    --stats "*" \
    --player_team_stats "*" \
    --opp_team_stats "*" \
    --calc_stats "*_score" \
    --current_extra "game_pos_*_rank" is_home odds_ou odds_spread \
    --target_calc_stats "*" \
    --target_stats asst blks d_reb ft_made o_reb fg_att pts tfg_att tfg_made turnovers \
    --hist_recent_games 5 --hist_recent_mode ma \
    --dask_inf_multi_season_mode processes --dask_tasks 3 \
    --slack --format parquet -f ${DEST}/nba_player.parquet

# team
dumpdata.sc $DB_FILE --seasons $SEASONS --no_players \
    --stats "*" \
    --opp_team_stats "*" \
    --current_extra elo_win_prob is_home "odds_*" \
    --current_opp_extra odds_ml \
    --target_stats pts win \
    --hist_recent_games 5 --hist_recent_mode ma \
    --progress --slack --format parquet -f ${DEST}/nba_team.parquet