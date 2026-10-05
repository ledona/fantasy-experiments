local utils = import 'utils.libsonnet';
{
  name: 'df-classic-misc',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
  },
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies:
    // baseline strategies
    [
      { name: 'baseline' },
    ] + [
      {
        name: 'baseline.' + score_data_type,
        score_data_type: score_data_type,
      }
      for score_data_type in ['ceiling', 'floor']
    ] +

    // drop inactive
    [
      {
        name: 'drop-inactive-players-%d' % time_off,
        gen_lineups_params: {
          exclude_for_time_off: time_off,
        },
      }
      for time_off in [2, 3, 4, 5]
    ] + [
      {
        name: 'drop-inactive-players-4.ceiling',
        score_data_type: 'ceiling',
        gen_lineups_params: {
          exclude_for_time_off: 4,
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
          exclude_teams_for_consec_epochs_played: 2,
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
        name: 'top-player-pivot:mlb-P:1.ceiling',
        sport_filter: 'mlb',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: 'P',
        max_bets_per_contest: 1,
        max_bets_per_epoch: 1,
      },
      {
        name: 'top-player-pivot:mlb-H:1.ceiling',
        sport_filter: 'mlb',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: 'C,1B,2B,3B,SS,LF,CF,RF,OF',
        max_bets_per_contest: 1,
        max_bets_per_epoch: 1,
      },
      {
        name: 'top-player-pivot:nfl-QB:1.ceiling',
        sport_filter: 'nfl',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: 'QB',
        max_bets_per_contest: 1,
        max_bets_per_epoch: 1,
      },
      {
        name: 'top-player-pivot:nfl-RB:1.ceiling',
        sport_filter: 'nfl',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: 'RB',
        max_bets_per_contest: 1,
        max_bets_per_epoch: 1,
      },
      {
        name: 'top-player-pivot:nfl-WRTE:1.ceiling',
        sport_filter: 'nfl',
        score_data_type: 'ceiling',
        multi_lineup_req_top_predicted_player: 'WR,TE',
        max_bets_per_contest: 1,
        max_bets_per_epoch: 1,
      },
    ] + [
      {
        name: 'top-player-pivot:mlb-P:%d' % bets,
        sport_filter: 'mlb',
        multi_lineup_req_top_predicted_player: 'P',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
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
        name: 'top-player-pivot:nfl-QB:%d' % bets,
        sport_filter: 'nfl',
        multi_lineup_req_top_predicted_player: 'QB',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] + [
      {
        name: 'top-player-pivot:nfl-RB:%d' % bets,
        sport_filter: 'nfl',
        multi_lineup_req_top_predicted_player: 'RB',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] + [
      {
        name: 'top-player-pivot:nfl-WRTE:%d' % bets,
        sport_filter: 'nfl',
        multi_lineup_req_top_predicted_player: 'WR,TE',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] + [
      {
        name: 'top-player-pivot:nba:%d' % bets,
        sport_filter: 'nba',
        multi_lineup_req_top_predicted_player: true,
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] + [
      {
        name: 'top-player-pivot:nhl-G:%d.%s' % [bets, utils.score_type_post(score_type)],
        score_data_type: score_type,
        sport_filter: 'nhl',
        multi_lineup_req_top_predicted_player: 'G',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
      for score_type in ['predicted', 'ceiling']
    ] + [
      {
        name: 'top-player-pivot:nhl-WC:%d' % bets,
        sport_filter: 'nhl',
        multi_lineup_req_top_predicted_player: 'W,C',
        max_bets_per_contest: bets,
        max_bets_per_epoch: bets,
      }
      for bets in [1, 2, 3]
    ] +

    // drop players against top team by position
    [
      {
        name: 'drop-players-against-top-team-vs-pos.%d.ceiling' % pctl,
        score_data_type: 'ceiling',
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
        gen_lineups_params: {
          exclude_player_for_opp_team_vs_pos_pctl: {
            '': pctl,
          },
        },
      }
      for pctl in [60, 70, 80, 90]
    ],
}
