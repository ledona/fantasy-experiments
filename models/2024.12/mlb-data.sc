#!/bin/bash

DB_FILE=${FANTASY_HOME}/mlb.hist.2008-2024.scored.db
DEST=/fantasy-isync/fantasy-modeling/2024.12/data
SEASONS="2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024"

# dump hitter data, apart from obvious targets, 
# the extra stat hitter_bases is also possible
dumpdata.sc --seasons $SEASONS --no_team --only_starters \
   --pos LF CF RF 1B 2B 3B SS C OF DH \
   --stats off_1b off_2b off_3b off_ab off_bb off_hbp off_hit \
      off_hr off_k off_pa "off_r*" "off_sac*" "off_sb*" \
   --player_team_stats off_bb off_hit off_hr off_pa off_rbi off_runs \
   --opp_team_stats errors "p_*" \
   --calc_stats "*_score" \
   --current_extra "hitter_*" indoors is_home \
      odds_ou odds_spread "team_o*" "team_s*" venue "weather_*"\
   --current_opp_extra "team_whip*" "sp*" \
   --hist_extra hitter_bases team_bases \
   --target_stats off_bb off_hit off_hr off_rbi off_runs off_sb \
   --target_calc_stats "*_score" \
   --hist_recent_games 5 --hist_recent_mode ma \
   --dask_mode processes --dask_tasks 4 \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_hitter.parquet

# dump pitchers
dumpdata.sc --seasons $SEASONS --no_team \
   --only_starters --pos P \
   --stats p_bb p_cg p_er p_hbp p_hits p_hr \
      p_ip p_k p_loss p_pc p_qs p_runs p_strikes p_win p_wp \
   --player_team_stats errors off_runs p_hold p_qs p_runs p_save win \
   --opp_team_stats errors off_ab off_hit off_hr off_k off_pa off_runs off_sb win \
   --calc_stats "*_score" \
   --current_extra elo_win_prob indoors is_home "odds_*" recent_player_win venue "weather_*" \
   --current_opp_extra odds_ml "team_hit*" "team_ops*" "team_slug*" \
   --hist_opp_extra team_bases \
   --target_calc_stats "*_score" \
   --target_stats p_bb p_hits p_hr p_ip p_k p_runs p_win \
   --hist_recent_games 5 --hist_recent_mode ma \
   --dask_mode processes --dask_tasks 6 \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_pitcher.parquet

# teams
TEAM_STATS="errors off_ab off_bb off_hit off_hr off_k off_pa off_rlob off_runs off_sb 
p_bb p_hits p_hold p_hr p_k p_pc p_qs p_runs p_save win"
TEAM_CURRENT_X="odds_ml sp_* team_h* team_ops* team_slug* team_whip*"
dumpdata.sc --seasons $SEASONS --no_player \
   --stats $TEAM_STATS \
   --opp_team_stats $TEAM_STATS \
   --current_extra $TEAM_CURRENT elo_win_prob indoors is_home \
      odds_ou odds_spread venue "weather_*" \
   --current_opp_extra $TEAM_CURRENT_X \
   --hist_extra team_bases \
   --hist_opp_extra team_bases \
   --target_stats off_runs win \
   --hist_recent_games 5 --hist_recent_mode ma \
   --dask_mode processes --dask_tasks 6 \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_team.parquet
