local utils = import 'utils.libsonnet';
{
  name: 'df-classic-mlb',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
    sport_filter: ['mlb'],
  },
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    { name: 'baseline' },
    {
      name: 'good-P-vs-bad-offense',
      gen_lineups_params: {
        exclude_below_ppg_pctl: { P: 60 },
        exclude_player_vs_team_predicted_pctl: { P: 25 },
      },
    },
  ] + [
    {
      name: 'current-recommended' + (if data_type != 'predicted' then ':' + data_type else ''),
      score_data_type: data_type,
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_predicted_pctl: 75,
        stackable_players_ppg_pctl: 50,
        exclude_below_ppg_pctl: { P: 70 },
        exclude_below_games_played: 2,
      },
    }
    for data_type in ['predicted', 'floor', 'ceiling']
  ] + [
    {
      name: 'batting-order-stack-%dx%d' % [stacks[1], stacks[0]] + utils.score_type_post(data_type),
      score_data_type: data_type,
      gen_lineups_params: {
        constraints: [
          ['mlb_bo_stack', { stack_size: stacks[0], stacks: stacks[1] }],
        ],
      },
    }
    for data_type in ['predicted', 'ceiling']
    for stacks in [[3, 2], [4, 1], [5, 1]]
  ] + [
    {
      name: 'vs-P-5.20.ceiling',
      score_data_type: 'ceiling',
      gen_lineups_params: {
        stacks: [5],
        stackable_teams_against_pos_ppg_pctl: ['P', 20],
      },
    },
    {
      name: 'vs-P-4.20.ceiling',
      score_data_type: 'ceiling',
      gen_lineups_params: {
        stacks: [4],
        stackable_teams_against_pos_ppg_pctl: ['P', 20],
      },
    },
    {
      name: 'vs-P-43.30.ceiling',
      score_data_type: 'ceiling',
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_against_pos_ppg_pctl: ['P', 30],
      },
    },
  ] + [
    {
      name: 'vs-P-5.%d' % pctl,
      gen_lineups_params: {
        stacks: [5],
        stackable_teams_against_pos_ppg_pctl: ['P', pctl],
      },
    }
    for pctl in [20, 30, 40]
  ] + [
    {
      name: 'vs-P-4.%d' % pctl,
      gen_lineups_params: {
        stacks: [4],
        stackable_teams_against_pos_ppg_pctl: ['P', pctl],
      },
    }
    for pctl in [10, 20, 30, 40]
  ] + [
    {
      name: 'vs-P-43.%d' % pctl,
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_against_pos_ppg_pctl: ['P', pctl],
      },
    }
    for pctl in [20, 30, 40]
  ] + [
    {
      name: 'prioritize-P',
      gen_lineups_params: {
        require_top_predicted_players: ['P', 2],
      },
    },
    {
      name: 'pitcher-budget:<18.5k.ceiling',
      score_data_type: 'ceiling',
      service_filter: ['draftkings'],
      gen_lineups_params: {
        group_cost_budget: { P: '<18500' },
      },
    },
    {
      name: 'pitcher-budget:>=19.5k',
      service_filter: ['draftkings'],
      gen_lineups_params: {
        group_cost_budget: { P: '>=19500' },
      },
    },
    {
      name: 'pitcher-budget:>=20k',
      service_filter: ['draftkings'],
      gen_lineups_params: {
        group_cost_budget: { P: '>=20000' },
      },
    },
    {
      name: 'pitcher-budget:<17.5k',
      service_filter: ['draftkings'],
      gen_lineups_params: {
        group_cost_budget: { P: '<17500' },
      },
    },
  ] + [
    {
      name: 'pitcher-budget:%s%sk' % [cond, budget],
      service_filter: ['draftkings'],
      gen_lineups_params: {
        group_cost_budget: { P: '%s%d' % [cond, budget * 1000] },
      },
    }
    for cond in ['<', '>=']
    for budget in [17, 18, 18.5, 19]
  ],
}
