// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
local utils = import 'utils.libsonnet';
{
  name: 'df-classic-nfl',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 2,
    max_bets_per_contest: 1,
    sport_filter: 'nfl',
    service_filter: 'draftkings',
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
        stacks: [2, 2],
        stackable_teams_predicted_pctl: 60,
        stackable_players_ppg_pctl: 65,
        exclude_below_ppg_pctl: { QB: 80 },
        exclude_for_cost: '<=2500',
        constraints: [
          ['stack_same_team_diff_pos', {
            position: 'QB',
            team_position: 'team.D',
            stack_positions: ['TE', 'WR'],
          }],
          ['opp_team', { team_position: 'team.D', position: 'team.D' }],
          ['max_team_players', { max_players: 2, positions: ['TE', 'WR'] }],
          ['max_team_players', { max_players: 1, positions: ['RB'] }],
          // 2026.11: bring-back, at least 1 WR/TE/RB from the QB's opponent (standard GPP game stack)
          ['opp_team', {
            position: 'QB',
            team_position: 'team.D',
            opp_positions: ['WR', 'TE', 'RB'],
            min_opp_players: 1,
            max_opp_players: null,
          }],
        ],
      },
    },
    // 2026.11: bring-back on its own, isolates its effect from the rest of rational.1
    {
      name: 'nfl-bring-back',
      gen_lineups_params: {
        constraints: [
          ['opp_team', {
            position: 'QB',
            team_position: 'team.D',
            opp_positions: ['WR', 'TE', 'RB'],
            min_opp_players: 1,
            max_opp_players: null,
          }],
        ],
      },
    },
    {
      name: 'rational.2:mini-slate',
      gen_lineups_params: {
        stacks: [2],
        stackable_teams_predicted_pctl: 75,
        exclude_for_cost: '<=1000',
        constraints: [
          ['stack_same_team_diff_pos', {
            position: 'QB',
            team_position: 'team.D',
            stack_positions: ['TE', 'WR'],
          }],
          ['opp_team', { team_position: 'team.D', position: 'team.D' }],
        ],
      },
    },
    // 2026.11: derived from rational.2:mini-slate (+5.9 on binding epochs) + cheap-WRTE (+7.1) (2026.10-2 NFL §5.1)
    {
      name: 'nfl-from-rational.2:QB-stack+cheap-WRTE',
      gen_lineups_params: {
        group_cost_budget: { 'WR,TE': 20000 },
        constraints: [
          ['stack_same_team_diff_pos', {
            position: 'QB',
            team_position: 'team.D',
            stack_positions: ['TE', 'WR'],
          }],
          ['opp_team', { team_position: 'team.D', position: 'team.D' }],
        ],
      },
    },
    {
      name: 'good-QB-vs-bad-defense',
      gen_lineups_params: {
        exclude_below_ppg_pctl: { QB: 60 },
        exclude_player_vs_team_predicted_pctl: { QB: 25 },
      },
    },
    {
      name: 'cheap-WRTE.gen<20k',
      gen_lineups_params: {
        group_cost_budget: { 'WR,TE': 20000 },
      },
    },
    {
      name: 'RBs-cap:<13K:ceiling',
      description: 'cap total spend on running backs',
      score_data_type: 'ceiling',
      gen_lineups_params: {
        group_cost_budget: { RB: '<13000' },
      },
    },
  ] + [
    // 2026.11: WR/TE spend x QB salary band x RB cap (2026.10-2 NFL cheap-WRTE +7.1, QB 6.5-7.5k +5.1, RB<16k mildly +)
    {
      name: 'nfl-combo:WRTE<%dk,QB-%s,RB-%s' % [wrte, if qb == null then 'any' else qb, if rb == null then 'any' else rb],
      gen_lineups_params: {
        group_cost_budget: if rb == null then { 'WR,TE': wrte * 1000 } else { 'WR,TE': wrte * 1000, RB: rb },
        slot_budget: if qb == null then null else { QB: qb },
      },
    }
    for wrte in [18, 19, 20, 21]
    for qb in [null, '6500-7500', '6000-8000']
    for rb in [null, '<16000']
    // WR,TE<20k alone duplicates cheap-WRTE.gen<20k
    if !(wrte == 20 && qb == null && rb == null)
  ] + [
    {
      name: 'prioritize-%s' % prioritization[0],
      gen_lineups_params: {
        require_top_predicted_players: prioritization,
      },
    }
    // RB kept as a sentinel for expensive-RB construction (2026.10-2 NFL §4)
    for prioritization in [['RB', 2], ['QB', 1], ['WR,TE', 2]]
  ] + [
    {
      name: 'QB-cap:%s%sk' % budget,
      gen_lineups_params: {
        slot_budget: { QB: '%s%d' % [budget[0], budget[1] * 1000] },
      },
    }
    for budget in [['<', 5], ['<', 5.5], ['<', 6], ['<', 6.5], ['<', 7], ['>=', 7]]
  ] + [
    {
      name: 'QB-cost-range:%g-%gk' % [budget_min, budget_min + range],
      gen_lineups_params: {
        slot_budget: { QB: '%d-%d' % [budget_min * 1000, (budget_min + range) * 1000] },
      },
    }
    for budget_min in [5, 5.5, 6, 6.5, 7]
    for range in [1, 1.5, 2]
    // 2026.11: QB upper bounds above 8k never bind (2026.10-2 NFL §5.3)
    if budget_min + range <= 8
  ] + [
    {
      name: 'RBs-cap:%s%sK' % budget,
      description: 'cap total spend on running backs',
      gen_lineups_params: {
        group_cost_budget: { RB: '%s%d' % [budget[0], budget[1] * 1000] },
      },
    }
    for budget in [['<', 12], ['<', 13], ['<', 14], ['<', 16], ['<', 18], ['<', 20], ['<', 22], ['>=', 22]]
  ] + [
    {
      local range = 3,
      name: 'RBs-range:%s-%sK' % [budget_min, budget_min + range],
      description: 'cap total spend on running backs',
      gen_lineups_params: {
        group_cost_budget: { RB: '%d-%d' % [(budget_min * 1000), (budget_min + range) * 1000] },
      },
    }
    // 2026.11: 16-19K kept as the only expensive-RB sentinel (2026.10-2 NFL §5.2)
    for budget_min in std.range(10, 16)
  ] + [
    // 2026.11: per-player cost floor, all binds ~44%/78% and WR,TE ~28%/61% of 2026.10-2 baseline lineups
    {
      name: 'cost-floor:<%gk' % floor,
      gen_lineups_params: { exclude_for_cost: '<%d' % (floor * 1000) },
    }
    for floor in [3, 3.5]
  ] + [
    {
      name: 'cost-floor:WR,TE<%gk' % floor,
      gen_lineups_params: { exclude_for_cost: { 'WR,TE': '<%d' % (floor * 1000) } },
    }
    for floor in [3.5, 4]
  ] + [
    // 2026.11: cap the number of expensive players (winner's-curse test); binds ~44%/17% of 2026.10-2 baseline lineups
    {
      name: 'max-expensive:%gk-max%d' % [pool[0] / 1000, pool[1]],
      gen_lineups_params: { cost_player_pool: ['>=%d' % pool[0], '<=%d' % pool[1]] },
    }
    for pool in [[7000, 1], [9000, 0]]
  ],
}
