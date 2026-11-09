// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
local utils = import 'utils.libsonnet';
{
  name: 'df-classic-nhl',
  params: {
    service_filter: 'draftkings',
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
    sport_filter: 'nhl',
  },
  starting_bankroll: 1000,
  allow_negative_bankroll: true,
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    {
      name: 'baseline%s' % utils.score_type_post(score_type),
      score_data_type: score_type,
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'rational.1',
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_predicted_pctl: 80,
        stackable_players_ppg_pctl: 65,
        exclude_below_ppg_pctl: { G: 80 },
        constraints: [
          ['opp_team', { position: 'G', opp_positions: ['W', 'LW', 'RW', 'C'] }],
        ],
      },
    },
    {
      name: 'rational.2',
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_predicted_pctl: 60,
        stackable_players_ppg_pctl: 55,
        exclude_below_ppg_pctl: { G: 60 },
        constraints: [
          ['opp_team', { position: 'G', opp_positions: ['W', 'LW', 'RW', 'C'] }],
        ],
      },
    },
    {
      // fallback for failure to solve but with a large number of games, so lossen exclusion but increase stackable threshold
      name: 'rational.3',
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_predicted_pctl: 75,
        stackable_players_ppg_pctl: 55,
        exclude_below_ppg_pctl: { G: 60 },
        constraints: [
          ['opp_team', { position: 'G', opp_positions: ['W', 'LW', 'RW', 'C'] }],
        ],
      },
    },
    {
      name: 'good-G-vs-bad-offense',
      gen_lineups_params: {
        exclude_below_ppg_pctl: { G: 60 },
        exclude_player_vs_team_predicted_pctl: { G: 25 },
      },
    },
    {
      name: 'prioritize-G',
      gen_lineups_params: {
        require_top_predicted_players: 'G',
      },
    },
  ] + [
    // 2026.11: goalie-skater correlation, at least N skaters from the goalie's team (G wins/saves track own skaters)
    {
      name: 'G-stack:%d' % size,
      gen_lineups_params: {
        constraints: [
          ['stack_same_team_diff_pos', { position: 'G', stack_positions: ['C', 'W', 'LW', 'RW', 'D'], stack_size: size }],
        ],
      },
    }
    for size in [2, 3]
  ] + [
    // 2026.11: goalie lockout alone (no skaters facing the goalie), until now only tested bundled in the rationals
    {
      name: 'G-lockout',
      gen_lineups_params: {
        constraints: [
          ['opp_team', { position: 'G', opp_positions: ['W', 'LW', 'RW', 'C'] }],
        ],
      },
    },
    {
      name: 'G-stack:3+G-lockout',
      gen_lineups_params: {
        constraints: [
          ['stack_same_team_diff_pos', { position: 'G', stack_positions: ['C', 'W', 'LW', 'RW', 'D'], stack_size: 3 }],
          ['opp_team', { position: 'G', opp_positions: ['W', 'LW', 'RW', 'C'] }],
        ],
      },
    },
  ] + [
    {
      name: 'current-recommended' + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_predicted_pctl: 80,
        stackable_players_ppg_pctl: 55,
        exclude_below_ppg_pctl: { G: 80 },
        exclude_for_cost: '<4000',
      },
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'dk-goalie:<%sk' % budget,
      gen_lineups_params: {
        slot_budget: { G: '<%d' % (budget * 1000) },
      },
    }
    for budget in [7, 7.5, 8]
  ] + [
    {
      local range = 1,
      name: 'dk-goalie:%s-%sk' % [budget_min, budget_min + range],
      gen_lineups_params: {
        slot_budget: { G: '%d-%d' % [(budget_min * 1000), ((budget_min + range) * 1000)] },
      },
    }
    for budget_min in [6, 6.5, 7, 7.5, 8]
  ] + [
    // 2026.11: sentinel set only
    {
      name: 'vs-G-%s.%d:%s' % [stack, g_pctl, ppg_pctl],
      gen_lineups_params: {
        stacks: stack,
        stackable_players_ppg_pctl: ppg_pctl,
        stackable_teams_against_pos_ppg_pctl: ['G', g_pctl],
      },
    }
    for stack in [[4], [5]]
    for ppg_pctl in [25, 45]
    for g_pctl in [20, 40]
  ] + [
    {
      // 2026.11: %g so 32.5k is named correctly (2026.10-2 NHL 2.1.c)
      name: 'S-budget:>=%gk' % budget,
      gen_lineups_params: {
        group_cost_budget: { 'W,C': '>=%d' % (budget * 1000) },
      },
    }
    for budget in [30, 32.5, 35, 37.5]
  ] + [
    // 2026.11: opp-vs-pos exclusion fine grid and per position group, alone and with 4/5 stacks
    // (2026.10-2 NHL drop-players-against-top-team-vs-pos.70 +5.3, gen-4/5 mildly positive)
    {
      name: 'opp-vs-pos:%s:stack-%s' % [opp[0], if stacks == null then 'none' else utils.stacks_to_str(stacks)],
      gen_lineups_params: {
        exclude_player_for_opp_team_vs_pos_pctl: opp[1],
        stacks: stacks,
      },
    }
    for opp in [
      ['all.65', { '': 65 }],
      ['all.70', { '': 70 }],
      ['all.75', { '': 75 }],
      ['W,C.70', { W: 70, C: 70 }],
      ['D.70', { D: 70 }],
      ['G.70', { G: 70 }],
    ]
    for stacks in [null, [4], [5]]
    // all.70 with no stack duplicates misc drop-players-against-top-team-vs-pos.70
    if !(opp[0] == 'all.70' && stacks == null)
  ] + [
    // 2026.11: per-player cost floor, all binds ~21%/74%, W,C ~49%, D ~41% of 2026.10-2 baseline lineups
    {
      name: 'cost-floor:<%gk' % floor,
      gen_lineups_params: { exclude_for_cost: '<%d' % (floor * 1000) },
    }
    for floor in [3, 3.5]
  ] + [
    {
      name: 'cost-floor:W,C<3.5k',
      gen_lineups_params: { exclude_for_cost: { 'W,C': '<3500' } },
    },
    {
      name: 'cost-floor:D<3.5k',
      gen_lineups_params: { exclude_for_cost: { D: '<3500' } },
    },
  ] + [
    // 2026.11: cap the number of expensive skaters (winner's-curse test); binds ~20%/14%/33% of 2026.10-2 baseline lineups
    {
      name: 'max-expensive:skaters-%gk-max%d' % [pool[0] / 1000, pool[1]],
      gen_lineups_params: { cost_player_pool: ['>=%d' % pool[0], '<=%d' % pool[1], 'W,C,D'] },
    }
    for pool in [[6500, 2], [7500, 1], [8500, 0]]
  ],
}
