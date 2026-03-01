#!/bin/bash

DB_FILE=${FANTASY_HOME}/nhl.hist.20072008-20242025.scored.db
DEST=/fantasy-isync/fantasy-modeling/2026.03/data
# not including 20072008, 20082009 because they don't have give|take aways
SEASONS='20092010 20102011 20112012 20122013 20132014 20142015 20152016 20162017 
20172018 20182019 20192020 20202021 20212022 20222023 20232024 20242025'
SKATER_STATS=("assist*" "fo*" "*away" "goal" "goal_pp" "goal_sh" 
    "goal_t" "goal_w" "hit" "p*" "shift" "shot*" "toi" "toi_ev" "toi_pp" "toi_sh")
GOALIE_STATS='giveaway goal_ag loss save toi win'
SHARED_CURRENT_X="is_home odds_ou odds_ml thfa elo_mov days_rest_team"
SHARED_CURRENT_OPP_X=("sg_*" "days_rest_team")
DASK_TASKS=7

# skater data
dumpdata.sc $DB_FILE --slack --seasons $SEASONS \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --no_teams --pos LW RW W C D \
    --stats "${SKATER_STATS[@]}" \
    --player_team_stats "*" --opp_team_stats "*" \
    --calc_stats "*_score" \
    --current_extra "game_pos_*_rank" skater_line days_rest_player $SHARED_CURRENT_X \
    --current_opp_extra "${SHARED_CURRENT_OPP_X[@]}" \
    --target_stats assist goal shot toi \
    --target_calc_stats "*_score" \
    --hist_recent_games 5 --hist_recent_mode m \
    --inf_nofail_cols "*shift*#20092010-20172018" \
    --format parquet -f ${DEST}/nhl_skater.parquet

# goalie data
dumpdata.sc $DB_FILE --seasons $SEASONS --slack \
    --dask_mode processes  --dask_tasks ${DASK_TASKS} \
    --no_teams \
    --pos G --starters \
    --stats $GOALIE_STATS \
    --player_team_stats "*" --opp_team_stats "*" \
    --calc_stats "*_score" \
    --current_extra elo_win_prob days_rest_player "g_save_pct*" $SHARED_CURRENT_X \
    --current_opp_extra "${SHARED_CURRENT_OPP_X[@]}" \
    --target_stats goal_ag save \
    --target_calc_stats "*_score" \
    --hist_recent_games 5 --hist_recent_mode m \
    --format parquet -f ${DEST}/nhl_goalie.parquet

# team data
dumpdata.sc $DB_FILE --seasons $SEASONS --slack \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --no_players \
    --stats "*" --opp_team_stats "*" \
    --current_extra elo_win_prob "sg_*" $SHARED_CURRENT_X \
    --current_opp_extra "${SHARED_CURRENT_OPP_X[@]}" \
    --target_stats goal win \
    --hist_recent_games 5 --hist_recent_mode m \
    --format parquet -f ${DEST}/nhl_team.parquet
