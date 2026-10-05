local utils = import 'utils.libsonnet';
{
  name: 'df-classic-nba',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
    sport_filter: ['nba'],
  },
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    {
      name: 'baseline',
    },
  ] + [
    {
      name: '3x2.T70.ppg25',
      gen_lineups_params: {
        stacks: [3, 2],
        stackable_teams_predicted_pctl: 70,
        exclude_below_ppg_pctl: 25,
      },
    },
    {
      name: '3x2x2.T50.ppg25',
      gen_lineups_params: {
        stacks: [3, 2, 2],
        stackable_teams_predicted_pctl: 50,
        exclude_below_ppg_pctl: 25,
      },
    },
  ] + [
    {
      name: '3x3.T70.ppg%d' % ppg_pctl,
      gen_lineups_params: {
        stacks: [3, 3],
        stackable_teams_predicted_pctl: 70,
        exclude_below_ppg_pctl: ppg_pctl,
      },
    }
    for ppg_pctl in [35, 40]
  ] + [
    {
      name: '3x3.T%d.ppg25' % pctl + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: {
        stacks: [3, 3],
        stackable_teams_predicted_pctl: pctl,
        exclude_below_ppg_pctl: 25,
      },
    }
    for pctl in [50, 60, 70]
    for score_type in ['predicted', 'ceiling']
  ],
}
