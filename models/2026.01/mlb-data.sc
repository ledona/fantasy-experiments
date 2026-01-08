#!/bin/bash

DB_FILE=${FANTASY_HOME}/mlb.hist.2008-2025.scored.db
DEST=/fantasy-isync/fantasy-modeling/2026.01/mlb.w.extras/data
SEASONS="2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"
# no more than 1 processes (dask tasks) per 4 GB of available RAM
DASK_TASKS=3

# dump hitter data, apart from obvious targets,
dumpdata.sc --seasons $SEASONS --no_team --only_starters \
   --pos LF CF RF 1B 2B 3B SS C OF DH \
   --stats off_1b off_2b off_3b off_hr off_hit \
      "off_r*" off_ab off_pa "off_sac*" "off_sb*" \
      off_bb off_k off_hbp \
      off_groundballs off_linedrives off_flyballs \
   --player_team_stats off_bb off_hit off_hr off_pa off_rbi off_runs \
      off_groundballs off_linedrives off_flyballs \
   --opp_team_stats errors "p_*" \
   --calc_stats "*_score" \
   --current_extra "hitter_*" \
      odds_ou odds_spread "team_ops*" "team_slug*" \
      indoors is_home venue "weather_*" \
   --current_opp_extra "team_whip*" "sp_*" \
   --hist_extra hitter_bases team_bases \
   --target_stats off_bb off_hit off_hr off_rbi off_runs off_sb \
   --target_calc_stats "*_score" \
   --hist_recent_games 5 --hist_recent_mode m \
   --dask_mode processes --dask_tasks $DASK_TASKS \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_hitter.parquet

# dump pitchers
dumpdata.sc --seasons $SEASONS --no_team \
   --only_starters --pos P \
   --stats p_outs p_pc p_bb p_k p_wp p_hbp p_strikes \
      p_cg p_qs p_win p_loss \
      p_groundballs p_linedrives p_flyballs \
      p_hits p_runs p_er p_hr \
   --player_team_stats errors \
      off_runs \
      p_hold p_qs p_runs "p_save*" win \
   --opp_team_stats off_hit off_hr off_runs \
      off_k off_bb off_pa off_sb \
      off_groundballs off_linedrives off_flyballs \
      win \
   --calc_stats "*_score" \
   --current_extra sp_hand elo_win_prob "odds_*" \
      indoors is_home venue "weather_*" \
   --current_opp_extra "team_hit*side*" "team_ops*" "team_slug*" \
   --hist_opp_extra team_bases \
   --target_calc_stats "*_score" \
   --target_stats p_bb p_hits p_hr p_outs p_k p_runs p_win \
   --hist_recent_games 5 --hist_recent_mode ma \
   --dask_mode processes --dask_tasks $DASK_TASKS \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_pitcher.parquet

# teams
TEAM_STATS="errors off_ab off_bb off_groundballs off_linedrives off_flyballs off_hit off_hr off_k off_pa off_rbi off_rlob off_runs off_sb 
p_bb p_hits p_hold p_hr p_k p_pc p_qs p_runs p_save p_save_blown win"
TEAM_CURRENT_X="odds_* venue weather_* sp_* team_hit*side* team_ops* team_slug* team_whip* elo_win_prob indoors is_home"
dumpdata.sc --seasons $SEASONS --no_player \
   --stats $TEAM_STATS \
   --opp_team_stats $TEAM_STATS \
   --current_extra $TEAM_CURRENT_X \
   --current_opp_extra $TEAM_CURRENT_X \
   --hist_extra team_bases \
   --hist_opp_extra team_bases \
   --target_stats off_runs win \
   --hist_recent_games 5 --hist_recent_mode m \
   --dask_mode processes --dask_tasks $DASK_TASKS \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_team.parquet
