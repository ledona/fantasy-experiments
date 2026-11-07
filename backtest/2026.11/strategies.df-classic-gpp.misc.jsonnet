// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
local utils = import 'utils.libsonnet';
{
  name: 'df-classic-misc',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
  },
  missing_inference_cols_threshold_fail: { mlb: 1 },
  starting_bankroll: 1000,
  allow_negative_bankroll: true,
  filepath: 'config/strategy/df-classic-gpp.json',
  // 2026.11: few ceiling variants for nba/mlb classic, ceiling lost to predicted (2026.10-2).
  // baseline:ceiling and one pivot:ceiling per sport stay as ceiling sentinels.
  local ceiling_sport_filter = '!nba,!mlb',
  strategies:
    // baseline strategies
    [
      {
        name: 'baseline%s' % utils.score_type_post(score_type),
        score_data_type: score_type,
      }
      for score_type in ['predicted', 'ceiling']
    ] +

    // 2026.11 sentinel for the pruned drop-inactive family (2026.10-2)
    [
      {
        name: 'drop-inactive-players-3',
        gen_lineups_params: {
          exclude_for_time_off: 3,
        },
      },
    ] +

    // drop low scoring players
    [
      {
        name: 'drop-low-predict-score-team-players:mlb',
        sport_filter: 'mlb',
        gen_lineups_params: {
          exclude_on_predicted_team_pctl: {
            'C,1B,2B,3B,SS,LF,CF,RF,OF': 45,
          },
        },
      },
      {
        name: 'drop-low-predict-score-team-players:nba',
        sport_filter: [
          'nba',
        ],
        gen_lineups_params: {
          exclude_on_predicted_team_pctl: {
            '': 45,
          },
        },
      },
    ] + [
      {
        name: 'drop-low-predict-score-team-players:nhl.%d' % pctl,
        sport_filter: 'nhl',
        gen_lineups_params: {
          exclude_on_predicted_team_pctl: {
            'W,C': pctl,
          },
        },
      }
      for pctl in [35, 45, 55]
    ] + [
      {
        name: 'drop-low-predict-score-team-players:nfl',
        sport_filter: [
          'nfl',
        ],
        gen_lineups_params: {
          exclude_on_predicted_team_pctl: {
            'QB,RB,WR,TE': 45,
          },
        },
      },
    ] +

    // drop tired players/teams
    [
      {
        name: 'avoid-tired-teams:nba|nhl',
        sport_filter: 'nba,nhl',
        gen_lineups_params: {
          // back to back
          exclude_teams_for_consec_epochs_played: 1,
        },
      },
    ] + [
      {
        name: 'avoid-tired-teams:mlb.%d' % epochs,
        sport_filter: 'mlb',
        gen_lineups_params: {
          exclude_teams_for_consec_epochs_played: epochs,
        },
      }
      for epochs in [7, 10, 13, 16]
    ] +

    // pivot players
    [
      {
        name: 'top-player-pivot:mlb-P:%d' % bets,
        sport_filter: 'mlb',
        multi_lineup_req_top_predicted_player: 'P',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] + [
      // 2026.11: ceiling sentinel for mlb classic
      {
        name: 'top-player-pivot:mlb-P:1:ceiling',
        sport_filter: 'mlb',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: 'P',
        max_bets_per_contest: 1,
        max_bets_per_epoch: 1,
      },
    ] + [
      {
        name: 'top-player-pivot:mlb-H:%d' % bets,
        sport_filter: 'mlb',
        multi_lineup_req_top_predicted_player: 'C,1B,2B,3B,SS,LF,CF,RF,OF',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] + [
      {
        name: 'top-player-pivot:nfl-QB:%d%s' % [bets, utils.score_type_post(score_type)],
        sport_filter: 'nfl',
        score_data_type: score_type,
        multi_lineup_req_top_predicted_player: 'QB',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [2, 3]
      for score_type in ['predicted', 'ceiling']
    ] + [
      {
        name: 'top-player-pivot:nfl-RB:%d%s' % [bets, utils.score_type_post(score_type)],
        sport_filter: 'nfl',
        score_data_type: score_type,
        multi_lineup_req_top_predicted_player: 'RB',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
      for score_type in ['predicted', 'ceiling']
    ] + [
      {
        name: 'top-player-pivot:nfl-WRTE:%d%s' % [bets, utils.score_type_post(score_type)],
        sport_filter: 'nfl',
        score_data_type: score_type,
        multi_lineup_req_top_predicted_player: 'WR,TE',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
      for score_type in ['predicted', 'ceiling']
    ] + [
      {
        name: 'top-player-pivot:nba:%d' % bets,
        sport_filter: 'nba',
        multi_lineup_req_top_predicted_player: true,
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      // 2026.11: N up to 5 (2026.10-2 NBA best-of-3 +12.9)
      for bets in [2, 3, 4, 5]
    ] + [
      // 2026.11: pivot on a position group instead of any player
      {
        name: 'top-player-pivot:nba-%s:3' % pos,
        sport_filter: 'nba',
        multi_lineup_req_top_predicted_player: pos,
        max_bets_per_contest: 3,
        max_bets_per_epoch: 3,
      }
      for pos in ['C', 'G', 'F']
    ] + [
      // 2026.11: ceiling sentinel for nba classic
      {
        name: 'top-player-pivot:nba:3:ceiling',
        sport_filter: 'nba',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: true,
        max_bets_per_contest: 3,
        max_bets_per_epoch: 3,
      },
    ] + [
      {
        name: 'top-player-pivot:nhl-WC:%d%s' % [bets, utils.score_type_post(score_type)],
        score_data_type: score_type,
        sport_filter: 'nhl',
        multi_lineup_req_top_predicted_player: 'W,C',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
      for score_type in ['predicted', 'ceiling']
    ] +

    // drop players against top team by position
    [
      {
        name: 'drop-players-against-top-team-vs-pos.%d:ceiling' % pctl,
        score_data_type: 'ceiling',
        sport_filter: ceiling_sport_filter,
        gen_lineups_params: {
          exclude_player_for_opp_team_vs_pos_pctl: {
            '': pctl,
          },
        },
      }
      for pctl in [70, 80]
    ] +
    [
      {
        name: 'drop-players-against-top-team-vs-pos.%d' % pctl,
        // 2026.11: nba only at 70 as a sentinel, lost at every pctl (2026.10-2 NBA §4)
        sport_filter: if pctl == 70 then null else '!nba',
        gen_lineups_params: {
          exclude_player_for_opp_team_vs_pos_pctl: {
            '': pctl,
          },
        },
      }
      for pctl in [60, 70, 80, 90]
    ],
}
