#!/bin/bash

DB_FILE=${FANTASY_HOME}/nfl.hist.2009-2023.scored.db
DEST=/fantasy-isync/fantasy-modeling/2024.12/data
SEASONS="2009 2010 2011 2012 2013 2014 2015 2016 2017 
2018 2019 2020 2021 2022 2023"


TEAM_STATS="def_fumble_recov def_int def_sacks def_tds op_* passing_yds 
pen_yds pens pts rushing_yds turnovers win"

# WR TE
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --pos WR TE --no_teams \
    --stats tds "receiving*" \
    --calc_stats "*_score_off" \
    --player_team_stats $TEAM_STATS \
    --opp_team_stats $TEAM_STATS \
    --current_extra venue is_home "game_pos_*_rank" \
    --target_stats receiving_rec receiving_yds \
    --target_calc_stats "*_score_off" \
    --hist_recent_games 3 --hist_recent_mode ma \
    --dask_mode processes --slack -f ${DEST}/nfl_WRTE.csv

# RB
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos RB \
    --stats tds "receiving*" "rushing*" \
    --calc_stats "*_score_off" \
    --player_team_stats $TEAM_STATS \
    --opp_team_stats $TEAM_STATS \
    --current_extra venue is_home "game_pos_*_rank" \
    --target_calc_stats "*_score_off" \
    --target_stats receiving_rec receiving_yds rushing_yds \
    --hist_recent_games 3 --hist_recent_mode ma \
    --progress --slack -f ${DEST}/nfl_RB.csv

# QB
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos QB \
    --stats tds "rushing*" "passing*" \
    --calc_stats "*_score_off" \
    --player_team_stats $TEAM_STATS \
    --opp_team_stats $TEAM_STATS \
    --current_extra venue is_home \
    --target_stats "passing_*" rushing_yds \
    --target_calc_stats "*_score_off" \
    --hist_recent_mode ma --hist_recent_games 3 \
    --progress --slack -f ${DEST}/nfl_QB.csv

# Kicker
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos K \
    --stats "kicking_f*" \
    --calc_stats "*_score_off" \
    --player_team_stats $TEAM_STATS \
    --opp_team_stats $TEAM_STATS \
    --current_extra venue is_home \
    --target_stats kicking_fgm \
    --target_calc_stats "*_score_off" \
    --hist_recent_mode ma --hist_recent_games 3 \
    --progress --slack -f ${DEST}/nfl_K.csv

# team defence and win
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_players \
    --stats $TEAM_STATS \
    --calc_stats "*_score_def" \
    --opp_team_stats $TEAM_STATS \
    --current_extra venue is_home \
    --target_calc_stats "*_score_def" \
    --target_stat pts win \
    --hist_recent_games 3 --hist_recent_mode ma \
    --progress --slack -f ${DEST}/nfl_team.csv