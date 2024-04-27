#!/bin/bash

DB_FILE=${FANTASY_HOME}/mlb_hist_20082023.scored.db
SEASONS="2015 2016 2017 2018 2019 2020 2021 2022 2023"
FANTASY_TARGETS="dk_score y_score"
HITTER_STATS="off_1b off_2b off_3b off_ab off_bb off_hbp off_hit 
off_hr off_k off_pa off_rbi off_rbi_w2 off_rlob off_runs off_sac 
off_sb off_sb_c "
OPP_P_EXTRAS="opp_p_hand opp_p_bb_recent opp_p_bb_std opp_p_hits_recent 
opp_p_hits_std opp_p_hr_recent opp_p_hr_std opp_p_k_recent 
opp_p_k_std opp_p_runs_recent opp_p_runs_std"
PITCHER_STATS="p_bb p_cg p_er p_hbp p_hits p_hr p_ibb 
p_ip p_k p_loss p_pc p_qs p_runs p_strikes p_win p_wp"

# dump hitter data
dumpdata.sc --seasons $SEASONS --only_starters \
   --pos LF CF RF 1B 2B 3B SS C OF DH PH --no_team \
   --stats $HITTER_STATS \
   --current_extra venue is_home hitting_side bo $OPP_P_EXTRAS \
   --opp_team_stats errors "p_*" --player_team_stats "off_*" \
   --target_calc_stats $FANTASY_TARGETS --target_stats off_hit off_runs \
   --hist_recent_games 5 --hist_recent_mode ma \
   --progress $DB_FILE --format parquet -f mlb_hitter.pq

# dump pitchers
dumpdata.sc --seasons $SEASONS --no_team \
   --only_starters --pos P \
   --stats $PITCHER_STATS \
   --current_extra venue is_home "hit_*_opp" p_hand $OPP_P_EXTRAS \
   --opp_team_stats "off_*" win --player_team_stats win "off_*" \
   --target_calc_stats $FANTASY_TARGETS --target_stats p_k p_ip p_hits \
   --hist_recent_games 5 --hist_recent_mode ma \
   --progress $DB_FILE --format parquet -f mlb_pitcher.pq

# teams
dumpdata.sc --seasons $SEASONS --no_player \
   --stats "*" --opp_team_stats "*" \
   --current_extra venue is_home "*_hand" "hit_*" $OPP_P_EXTRAS \
   --target_stats off_runs win \
   --hist_recent_games 5 --hist_recent_mode ma \
   --progress $DB_FILE --format parquet -f mlb_team.pq
