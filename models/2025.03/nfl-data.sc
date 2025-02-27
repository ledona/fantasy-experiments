#!/bin/bash

DB_FILE=${FANTASY_HOME}/nfl.hist.2009-2024.scored.db
DEST=/fantasy-isync/fantasy-modeling/2025.03/data
SEASONS="2009 2010 2011 2012 2013 2014 2015 2016 2017 
2018 2019 2020 2021 2022 2023 2024"
TEAM_STATS="op_pts op_turnovers op_yds pen* pts turnovers yds"
OWN_TEAM_STATS="$TEAM_STATS passing_yds rushing_yds"
OPP_TEAM_STATS="$TEAM_STATS def_fumble_recov def_int def_sacks op_passing_yds op_rushing_yds"
CURRENT_X="indoors is_home elo_win_prob odds_ou odds_spread venue weather*"

# WR TE
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos WR TE \
    --stats fumbles_lost "receiving*" \
    --player_team_stats $OWN_TEAM_STATS \
    --opp_team_stats $OPP_TEAM_STATS  \
    --calc_stats "*_score_off" \
    --current_extra "game_pos_*_rank" $CURRENT_X \
    --target_stats receiving_rec receiving_targets receiving_tds receiving_yds \
    --target_calc_stats "*_score_off" \
    --hist_recent_games 3 --hist_recent_mode m \
    --dask_mode processes --slack -f ${DEST}/nfl_WRTE.csv

# RB
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos RB \
    --stats fumbles_lost "receiving*" "rushing*" tds \
    --player_team_stats $OWN_TEAM_STATS \
    --opp_team_stats $OPP_TEAM_STATS  \
    --calc_stats "*_score_off" \
    --current_extra "game_pos_*_rank" $CURRENT_X \
    --target_stats receiving_rec receiving_targets receiving_yds \
        rushing_att rushing_tds rushing_yds tds \
    --target_calc_stats "*_score_off" \
    --hist_recent_games 3 --hist_recent_mode m \
    --progress --slack -f ${DEST}/nfl_RB.csv

# QB
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos QB \
    --stats fumbles_lost "passing*" "rushing*" starting_qb tds \
    --player_team_stats $OWN_TEAM_STATS \
    --opp_team_stats $OPP_TEAM_STATS  \
    --calc_stats "*_score_off" \
    --current_extra $CURRENT_X \
    --target_stats "passing_*" rushing_att rushing_tds rushing_yds tds \
    --target_calc_stats "*_score_off" \
    --hist_recent_mode m --hist_recent_games 3 \
    --progress --slack -f ${DEST}/nfl_QB.csv

# Kicker
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_teams --pos K \
    --stats "kicking_f*" \
    --player_team_stats $TEAM_STATS \
    --opp_team_stats $TEAM_STATS \
    --calc_stats "*_score_off" \
    --current_extra $CURRENT_X \
    --target_stats kicking_fgm \
    --target_calc_stats "*_score_off" \
    --hist_recent_mode m --hist_recent_games 3 \
    --progress --slack -f ${DEST}/nfl_K.csv

# team defense and win
dumpdata.sc ${DB_FILE} \
    --seasons $SEASONS --no_players \
    --stats "*" \
    --opp_team_stats "*" \
    --calc_stats "*_score_def" \
    --current_extra $CURRENT_X odds_ml \
    --current_opp_extra odds_ml \
    --target_stat pts win \
    --target_calc_stats "*_score_def" \
    --hist_recent_games 3 --hist_recent_mode m \
    --progress --slack -f ${DEST}/nfl_team.csv