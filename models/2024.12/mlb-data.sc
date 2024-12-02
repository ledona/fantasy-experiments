#!/bin/bash

DB_FILE=${FANTASY_HOME}/mlb.hist.20082024.scored.db
DEST=/fantasy-isync/fantasy-modeling/2024.12/data
SEASONS="2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024"



HITTER_STATS="off_1b off_2b off_3b off_ab off_bb off_hbp off_hit 
off_hr off_k off_pa off_rbi* off_rlob off_runs off_sac* off_sb*"
PITCHER_STATS="p_bb p_cg p_er p_hbp p_hits p_hr 
p_ip p_k p_loss p_pc p_qs p_runs p_strikes p_win p_wp"

# dump hitter data
dumpdata.sc --seasons $SEASONS --no_team --only_starters \
   --pos LF CF RF 1B 2B 3B SS C OF DH \
   --stats $HITTER_STATS \
   --calc_stats "*_score" \
   --current_extra "team_*" "hitter_*" sp_hand venue is_home \
   --current_opp_extra "team_whip*" "sp*" \
   --hist_extra hitter_bases team_bases \
   --hist_opp_extra team_bases \
   --opp_team_stats errors "p_*" --player_team_stats "off_*" \
   --target_calc_stats $FANTASY_TARGETS --target_stats off_hit off_runs \
   --hist_recent_games 5 --hist_recent_mode ma \
   --dask_mode processes --dask_tasks 4 \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_hitter.parquet

# dump pitchers
dumpdata.sc --seasons $SEASONS --no_team \
   --only_starters --pos P \
   --stats $PITCHER_STATS \
   --calc_stats "*_score" \
   --current_extra "team_ops*" "sp_*" "*whip*" is_home venue \
   --current_opp_extra team_bases "team_hit*" "team_ops*" "team_slug*" \
   --hist_opp_extra team_bases \
   --opp_team_stats "off_*" win \
   --player_team_stats "off_*" win \
   --target_calc_stats $FANTASY_TARGETS --target_stats p_k p_ip p_hits \
   --hist_recent_games 5 --hist_recent_mode ma \
   --slack $DB_FILE --format parquet -f ${DEST}/mlb_pitcher.parquet

# teams
dumpdata.sc --seasons $SEASONS --no_player \
   --stats "*" \
   --opp_team_stats "*" \
   --current_extra venue is_home "team_*" "sp_*" \
   --current_opp_extra "team_*" "sp_*" \
   --hist_extra team_bases \
   --hist_opp_extra team_bases \
   --target_stats off_runs win \
   --hist_recent_games 5 --hist_recent_mode ma \
   --progress --slack $DB_FILE --format parquet -f ${DEST}/mlb_team.parquet
