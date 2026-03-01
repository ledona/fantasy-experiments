#!/bin/bash

DB_FILE=${FANTASY_HOME}/nfl.hist.2009-2025.scored.db
DEST=/fantasy-isync/fantasy-modeling/2026.03/data
SEASONS="2009 2010 2011 2012 2013 2014 2015 2016 2017 
2018 2019 2020 2021 2022 2023 2024 2025"
TEAM_STATS="op_pts op_turnovers op_yds pen* pts turnovers yds"
OWN_TEAM_STATS="$TEAM_STATS passing_yds rushing_yds"
OPP_TEAM_STATS="$TEAM_STATS def_fumble_recov def_int def_sacks op_passing_yds op_rushing_yds"
CURRENT_X="indoors is_home elo_mov odds_ou odds_spread venue thfa pf weather*"
DASK_TASKS=10

# WR TE
dumpdata.sc ${DB_FILE} \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --seasons $SEASONS --no_teams --pos WR TE \
    --stats fumbles_lost "receiving*" \
    --player_team_stats $OWN_TEAM_STATS \
    --opp_team_stats $OPP_TEAM_STATS  \
    --calc_stats "*_score_off" \
    --current_extra "game_pos_*_rank" $CURRENT_X \
    --current_opp_extra days_rest_team \
    --target_stats receiving_rec receiving_targets receiving_tds receiving_yds \
    --target_calc_stats "*_score_off" \
    --inf_nofail_cols extra:thfa#2009 \
    --hist_recent_games 3 --hist_recent_mode m \
    --slack -f ${DEST}/nfl_WRTE.csv

# RB
dumpdata.sc ${DB_FILE} \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --seasons $SEASONS --no_teams --pos RB \
    --stats fumbles_lost "receiving*" "rushing*" tds \
    --player_team_stats $OWN_TEAM_STATS \
    --opp_team_stats $OPP_TEAM_STATS  \
    --calc_stats "*_score_off" \
    --current_extra "game_pos_*_rank" $CURRENT_X \
    --current_opp_extra days_rest_team \
    --target_stats receiving_rec receiving_targets receiving_yds \
        rushing_att rushing_tds rushing_yds tds \
    --target_calc_stats "*_score_off" \
    --inf_nofail_cols extra:thfa#2009 \
    --hist_recent_games 3 --hist_recent_mode m \
    --slack -f ${DEST}/nfl_RB.csv

# QB: starting_qb is included so that data preprocessing can filter for starters before
#     training
dumpdata.sc ${DB_FILE} \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --seasons $SEASONS --no_teams --pos QB \
    --stats fumbles_lost "passing*" "rushing*" tds \
    --player_team_stats $OWN_TEAM_STATS \
    --opp_team_stats $OPP_TEAM_STATS  \
    --calc_stats "*_score_off" \
    --current_extra $CURRENT_X \
    --current_opp_extra days_rest_team \
    --target_stats "passing_*" rushing_att rushing_tds rushing_yds tds starting_qb \
    --target_calc_stats "*_score_off" \
    --hist_recent_mode m --hist_recent_games 3 \
    --inf_nofail_cols extra:thfa#2009 "stat:rushing_twoptm:*" \
    --slack -f ${DEST}/nfl_QB.csv

# Kicker
dumpdata.sc ${DB_FILE} \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --seasons $SEASONS --no_teams --pos K \
    --stats "kicking_f*" \
    --player_team_stats $TEAM_STATS \
    --opp_team_stats $TEAM_STATS \
    --current_extra $CURRENT_X \
    --target_stats kicking_fgm \
    --hist_recent_mode m --hist_recent_games 3 \
    --inf_nofail_cols extra:thfa#2009 \
    --slack -f ${DEST}/nfl_K.csv

# team defense and win
dumpdata.sc ${DB_FILE} \
    --dask_mode processes --dask_tasks ${DASK_TASKS} \
    --seasons $SEASONS --no_players \
    --stats "*" \
    --opp_team_stats "*" \
    --calc_stats "*_score_def" \
    --current_extra $CURRENT_X odds_ml days_rest_team elo_win_prob \
    --current_opp_extra odds_ml days_rest_team \
    --target_stat pts win \
    --target_calc_stats "*_score_def" \
    --hist_recent_games 3 --hist_recent_mode m \
    --inf_nofail_cols "stat:def_safety*" extra:thfa#2009 \
    --slack -f ${DEST}/nfl_team.csv