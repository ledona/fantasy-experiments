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
      name: 'baseline',
    },
    {
      name: 'good-G-vs-bad-offense',
      gen_lineups_params: {
        exclude_below_ppg_pctl: { G: 60 },
        exclude_player_vs_team_predicted_pctl: { G: 25 },
      },
    },
    {
      name: 'vs-G-43.20',
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_against_pos_ppg_pctl: ['G', 20],
      },
    },
    {
      name: 'prioritize-G',
      gen_lineups_params: {
        require_top_predicted_players: 'G',
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
    for budget in [7, 7.5, 8, 8.5]
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
    {
      name: 'vs-G-%d.%d:%s%s' % [stack, pctl, ppg_pctl, utils.score_type_post(score_type)],
      score_data_type: score_type,
      gen_lineups_params: {
        stacks: [stack],
        stackable_players_ppg_pctl: ppg_pctl,
        stackable_teams_against_pos_ppg_pctl: ['G', pctl],
      },
    }
    for stack in [4, 5]
    for ppg_pctl in [null, 10, 25, 40]
    for pctl in [10, 20, 30, 40]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'vs-G-43.%d:45' % pctl,
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_teams_against_pos_ppg_pctl: ['G', pctl],
        stackable_players_ppg_pctl: 45,
      },
    }
    for pctl in [10, 20, 30, 40]
  ] + [
    {
      name: 'S-budget:>=%dk' % budget,
      gen_lineups_params: {
        group_cost_budget: { 'W,C': '>=%d' % (budget * 1000) },
      },
    }
    for budget in [17, 15, 13]
  ] + [
    {
      name: 'stack-vs-crappy-5:nhl',
      gen_lineups_params: {
        stackable_teams_against_pos_ppg_pctl: ['W,C,D', 75],
        stacks: [5],
      },
    },
    {
      name: 'stack-vs-crappy-5:nhl.ceiling',
      score_data_type: 'ceiling',
      gen_lineups_params: {
        stackable_teams_against_pos_ppg_pctl: ['W,C,D', 75],
        stacks: [5],
      },
    },
    {
      name: 'stack-vs-crappy-43.75:nhl',
      gen_lineups_params: {
        stackable_teams_against_pos_ppg_pctl: ['W,C,D', 75],
        stacks: [4, 3],
      },
    },
    {
      name: 'stack-vs-crappy-43.60:nhl',
      gen_lineups_params: {
        stackable_teams_against_pos_ppg_pctl: ['W,C,D', 60],
        stacks: [4, 3],
      },
    },
  ],
}
