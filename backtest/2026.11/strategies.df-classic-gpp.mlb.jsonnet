// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
{
  name: 'df-classic-mlb',
  params: {
    service_filter: 'draftkings',
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
    sport_filter: 'mlb',
  },
  missing_inference_cols_threshold_fail: 1,
  starting_bankroll: 1000,
  allow_negative_bankroll: true,
  filepath: 'config/strategy/df-classic-gpp.json',
  // pitcher can't face any of the lineup's hitters
  local no_hitters_vs_p = ['opp_team', {
    position: 'P',
    opp_positions: 'C,1B,2B,3B,SS,LF,CF,RF,OF,DH',
    max_opp_players: 0,
  }],
  strategies:
    [
      {
        name: 'baseline',
      },
    ] + [
      // 2026.11: rational-1..3 primary stack must be near-consecutive in batting order (standard practice)
      {
        name: 'rational-1',
        gen_lineups_params: {
          stacks: [5, 2],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 50, '': 20 },
          exclude_below_games_played: { '': 3, P: 1 },
          constraints: [
            no_hitters_vs_p,
            ['mlb_bo_stack', { stack_size: 5, max_bo_distance: 5 }],
          ],
        },
      },
      {
        name: 'rational-2',
        gen_lineups_params: {
          stacks: [4, 3],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 50, '': 20 },
          exclude_below_games_played: { '': 3, P: 1 },
          constraints: [
            no_hitters_vs_p,
            ['mlb_bo_stack', { stack_size: 4, max_bo_distance: 4 }],
          ],
        },
      },
      {
        name: 'rational-3',
        gen_lineups_params: {
          stacks: [3, 2],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 50, '': 20 },
          exclude_below_games_played: { '': 3, P: 1 },
          constraints: [
            no_hitters_vs_p,
            ['mlb_bo_stack', { stack_size: 3, max_bo_distance: 3 }],
          ],
        },
      },
      // 2026.11: derived from 2026.10-2 rational-2 (+5.8 on fired epochs, [4,3] lost ~36% of epochs to explosion)
      {
        name: 'mlb-from-rational-2:[4,2]',
        gen_lineups_params: {
          stacks: [4, 2],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 50, '': 20 },
          exclude_below_games_played: { '': 3, P: 1 },
          constraints: [no_hitters_vs_p],
        },
      },
      {
        name: 'mlb-from-rational-2:[5]',
        gen_lineups_params: {
          stacks: [5],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 50, '': 20 },
          exclude_below_games_played: { '': 3, P: 1 },
          constraints: [no_hitters_vs_p],
        },
      },
      {
        name: 'mlb-from-rational-2:[4,2]+prioritize-P',
        gen_lineups_params: {
          stacks: [4, 2],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 50, '': 20 },
          exclude_below_games_played: { '': 3, P: 1 },
          require_top_predicted_players: ['P', 2],
          constraints: [no_hitters_vs_p],
        },
      },
    ] + [
      {
        name: 'rational-4:for mini slate (2 games)',
        gen_lineups_params: {
          stacks: [3],
          stackable_teams_predicted_pctl: 50,
          exclude_below_ppg_pctl: { P: 25, '': 20 },
          exclude_below_games_played: 1,
          constraints: [
            ['opp_team', {
              position: 'P',
              opp_positions: 'C,1B,2B,3B,SS,LF,CF,RF,OF,DH',
              max_opp_players: 2,
            }],
            // 2026.11: near-consecutive batting-order stack (standard practice)
            ['mlb_bo_stack', { stack_size: 3, max_bo_distance: 3 }],
          ],
        },
      },
      {
        name: 'good-P-vs-bad-offense',
        gen_lineups_params: {
          exclude_below_ppg_pctl: { P: 60 },
          exclude_player_vs_team_predicted_pctl: { P: 25 },
        },
      },
    ] + [
      {
        name: 'current-recommended',
        gen_lineups_params: {
          stacks: [4, 3],
          stackable_teams_predicted_pctl: 75,
          stackable_players_ppg_pctl: 50,
          exclude_below_ppg_pctl: { P: 70 },
          exclude_below_games_played: 2,
        },
      },
    ] + [
      {
        name: 'batting-order-stack-%d.%d' % [size, max_dist],
        gen_lineups_params: {
          constraints: [
            ['mlb_bo_stack', { stack_size: size, max_bo_distance: max_dist }],
          ],
        },
      }
      for size in std.range(3, 5)
      for max_dist in std.range(size - 1, (9 - std.ceil(9 / size)))
      // 2026.11: 3.6 and 4.6 dropped, identical to 3.5 / gen-4 (2026.10-2 MLB §5.3)
      if !std.member([[3, 6], [4, 6]], [size, max_dist])
    ] + [
      {
        name: 'vs-P-%s.%d' % [stacks, pctl],
        gen_lineups_params: {
          stacks: stacks,
          stackable_teams_against_pos_ppg_pctl: ['P', pctl],
        },
      }
      for pctl in [10, 20, 30, 40]
      for stacks in [[5], [4], [4, 3]]
    ] + [
      {
        name: 'prioritize-P',
        gen_lineups_params: {
          require_top_predicted_players: ['P', 2],
        },
      },
    ] + [
      // 2026.11: factorial of the three positive single filters (2026.10-2 MLB §4: prioritize-P,
      // avoid-tired-teams, drop-low-predict-score-team-players)
      {
        name: 'mlb-combo:P%d,tired-%d,team-pctl-%s' % [p_count, tired, if team_pctl == null then 'none' else team_pctl],
        gen_lineups_params: {
          require_top_predicted_players: ['P', p_count],
          exclude_teams_for_consec_epochs_played: tired,
          exclude_on_predicted_team_pctl: if team_pctl == null then null else { 'C,1B,2B,3B,SS,LF,CF,RF,OF': team_pctl },
        },
      }
      for p_count in [1, 2]
      for tired in [8, 10, 12]
      for team_pctl in [null, 35, 45, 55]
    ] + [
      {
        name: 'pitcher-budget:>=19.5k',
        gen_lineups_params: {
          group_cost_budget: { P: '>=19500' },
        },
      },
      {
        name: 'pitcher-budget:>=20k',
        gen_lineups_params: {
          group_cost_budget: { P: '>=20000' },
        },
      },
      {
        name: 'pitcher-budget:<17.5k',
        gen_lineups_params: {
          group_cost_budget: { P: '<17500' },
        },
      },
    ] + [
      {
        name: 'pitcher-budget:%s%sk' % [cond, budget],
        gen_lineups_params: {
          group_cost_budget: { P: '%s%d' % [cond, budget * 1000] },
        },
      }
      for cond in ['<', '>=']
      for budget in [17, 18, 18.5, 19]
    ] + [
      {
        local range = 3,
        name: 'pitcher-budget:%s-%sk' % [budget_min, budget_min + range],
        gen_lineups_params: {
          group_cost_budget: { P: '%s-%s' % [budget_min * 1000, (budget_min + range) * 1000] },
        },
      }
      for budget_min in std.range(13, 20)
    ] + [
      // 2026.11: per-player cost floor, all players binds ~48%/80% and P ~22%/55% of 2026.10-2 baseline lineups
      {
        name: 'cost-floor:<%gk' % floor,
        gen_lineups_params: { exclude_for_cost: '<%d' % (floor * 1000) },
      }
      for floor in [2.5, 3]
    ] + [
      {
        name: 'cost-floor:P<%gk' % floor,
        gen_lineups_params: { exclude_for_cost: { P: '<%d' % (floor * 1000) } },
      }
      for floor in [7.5, 8.5]
    ] + [
      // 2026.11: cap the number of expensive hitters (winner's-curse test); binds ~46%/60%/29% of 2026.10-2 baseline lineups
      {
        name: 'max-expensive:hitters-%gk-max%d' % [pool[0] / 1000, pool[1]],
        gen_lineups_params: { cost_player_pool: ['>=%d' % pool[0], '<=%d' % pool[1], 'C,1B,2B,3B,SS,LF,CF,RF,OF'] },
      }
      for pool in [[5000, 1], [5500, 0], [6000, 0]]
    ],
}
