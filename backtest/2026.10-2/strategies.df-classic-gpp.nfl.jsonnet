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
      name: 'baseline',
    },
    {
      name: 'rational.1',
      gen_lineups_params: {
        stacks: [3, 2],
        stackable_teams_predicted_pctl: 65,
        stackable_players_ppg_pctl: 65,
        exclude_below_ppg_pctl: { QB: 80 },
        exclude_for_cost: '<=2500',
        constraints: [
          ['stack_same_team_diff_pos', {
            position: 'QB',
            team_position: 'team.D',
            stack_positions: ['TE', 'WR'],
            stack_size: 2,
          }],
          ['no_opp_team', { team_position: 'team.D', position: 'team.D' }],
          ['max_team_players', { max_players: 2, positions: ['TE', 'WR'] }],
          ['max_team_players', { max_players: 1, positions: ['RB'] }],
        ],
      },
    },
    {
      name: 'rational.2',
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
          ['no_opp_team', { team_position: 'team.D', position: 'team.D' }],
          ['max_team_players', { max_players: 2, positions: ['TE', 'WR'] }],
          ['max_team_players', { max_players: 1, positions: ['RB'] }],
        ],
      },
    },
    {
      name: 'rational.3:mini-slate-1',
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
          ['no_opp_team', { team_position: 'team.D', position: 'team.D' }],
        ],
      },
    },
    {
      name: 'rational.4:mini-slate-2',
      gen_lineups_params: {
        // needed for mini slates (e.g. 4 or 2 games in the playoffs
        stacks: 2,
        stackable_teams_predicted_pctl: 75,
        exclude_for_cost: '<=1000',
        constraints: [
          ['stack_same_team_diff_pos', {
            position: 'QB',
            team_position: 'team.D',
            stack_positions: ['TE', 'WR'],
          }],
          ['no_opp_team', { team_position: 'team.D', position: 'team.D' }],
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
    {
      name: 'prioritize-%s' % prioritization[0],
      gen_lineups_params: {
        require_top_predicted_players: prioritization,
      },
    }
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
    for budget_min in [5, 5.5, 6, 6.5, 7, 7.5, 8]
    for range in [1, 2, 3]
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
    for budget_min in std.range(10, 20)
  ],
}
