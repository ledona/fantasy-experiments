local utils = import 'utils.libsonnet';
{
  name: 'df-classic-nfl',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 2,
    max_bets_per_contest: 1,
    sport_filter: 'nfl',
    service_filter: ['draftkings'],
  },
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    {
      name: 'baseline',
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
      name: 'RBs-cap:<13K.ceiling',
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
        group_cost_budget: { QB: '%s%d' % [budget[0], budget[1] * 1000] },
      },
    }
    for budget in [['<', 5], ['<', 5.5], ['<', 6], ['<', 6.5], ['<', 7], ['>=', 7]]
  ] + [
    {
      name: 'RBs-cap:%s%sK' % budget,
      description: 'cap total spend on running backs',
      gen_lineups_params: {
        group_cost_budget: { RB: '%s%d' % [budget[0], budget[1] * 1000] },
      },
    }
    for budget in [['<', 12], ['<', 13], ['<', 14], ['<', 16], ['<', 18], ['<', 20], ['<', 22], ['>=', 22]]
  ],
}
