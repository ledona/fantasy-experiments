#!/bin/bash

DB_FILE=${FANTASY_HOME}/nhl_hist_20072008-20222023.scored.db
# line stat is only available for skaters after 20152016
SEASONS='20152016 20162017 20172018 20182019 20192020 20202021 20212022 20222023'
SKATER_STATS=("assist*" "fo*" "*away" "goal" "goal_pp" "goal_sh" 
    "goal_t" "goal_w" "hit" line "p*" "shot*" "toi_ev" "toi_pp" "toi_sh")
GOALIE_STATS='goal_ag loss save toi_g win'

# skater data
dumpdata.sc --seasons $SEASONS --progress \
    $DB_FILE --no_teams \
    --pos LW RW W C D \
    --stats "${SKATER_STATS[@]}" --current_extra is_home "opp*" \
    --target_stats shot assist goal --target_calc_stats "*" \
    --player_team_stats "*" --opp_team_stats "*" \
    --hist_recent_games 5 --hist_recent_mode ma \
    --format parquet -f nhl_skater.pq

# goalie data
dumpdata.sc --seasons $SEASONS --progress \
    $DB_FILE --no_teams \
    --pos G --starters \
    --stats $GOALIE_STATS --current_extra is_home "opp*" \
    --target_calc_stats "*" --target_stats goal_ag save \
    --player_team_stats "*" --opp_team_stats "*" \
    --hist_recent_games 5 --hist_recent_mode ma \
    --format parquet -f nhl_goalie.pq

# team data
dumpdata.sc --seasons $SEASONS --progress \
    $DB_FILE --no_players \
    --stats "*" --target_stats goal win --current_extra is_home "opp*" \
    --opp_team_stats "*" \
    --hist_recent_games 5 --hist_recent_mode ma \
    --format parquet -f nhl_team.pq
