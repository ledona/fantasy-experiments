#!/bin/bash

DB_FILE=${FANTASY_HOME}/nba.hist.20082009-20242025.scored.db
DEST=/fantasy-isync/fantasy-modeling/2026.03/data
SEASONS="20082009 20092010 20102011 20112012 20122013 20132014 20142015 
20152016 20162017 20172018 20182019 20192020 20202021 20212022 20222023 
20232024 20242025"
DASK_TASKS=10
SHARED_X="is_home odds_ou odds_spread elo_mov thfa days_rest_team"

# player
dumpdata.sc $DB_FILE --seasons $SEASONS --no_teams \
    --stats asst blks "*_reb" "f*" "p*" stls "t*" \
    --player_team_stats "*" \
    --opp_team_stats "*" \
    --calc_stats "*_score" \
    --current_extra starter days_rest_player $SHARED_X \
    --current_opp_extra days_rest_team \
    --target_calc_stats "*" \
    --target_stats asst blks d_reb ft_made o_reb fg_att pts tfg_att tfg_made turnovers \
    --inf_nofail_cols extra:thfa#20082009 \
    --hist_recent_games 5 --hist_recent_mode m \
    --dask_inf_multi_season_mode processes --dask_tasks $DASK_TASKS \
    --slack --format parquet -f ${DEST}/nba_player.parquet

# team
dumpdata.sc $DB_FILE --seasons $SEASONS --no_players \
    --stats "*" \
    --opp_team_stats "*" \
    --current_extra elo_win_prob $SHARED_X \
    --current_opp_extra days_rest_team \
    --target_stats pts win \
    --inf_nofail_cols extra:thfa#20082009 \
    --hist_recent_games 5 --hist_recent_mode m \
    --dask_inf_multi_season_mode processes --dask_tasks $DASK_TASKS \
    --slack --format parquet -f ${DEST}/nba_team.parquet