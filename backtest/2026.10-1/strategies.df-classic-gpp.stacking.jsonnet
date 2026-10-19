local utils = import 'utils.libsonnet';
{
  name: 'df-classic-stacking',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
  },
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    { name: 'baseline' },
    {
      name: 'player-pctl-22.40.ceiling',
      score_data_type: 'ceiling',
      sport_filter: '!nfl',
      gen_lineups_params: {
        stacks: [2, 2],
        stackable_players_ppg_pctl: 40,
      },
    },
    {
      name: 'top-games-43:1.ceiling',
      sport_filter: 'nhl,nba',
      score_data_type: 'ceiling',
      multi_lineup_stack_top_games: [4, 3, 65],
      max_bets_per_contest: 1,
      max_bets_per_epoch: 1,
    },
  ] + [
    {
      name: 'filter-%s' % utils.stacks_to_str(stacks),
      stack_filter: stacks,
      sport_filter: 'mlb',
    }
    for stacks in [[4], [3, 3]]
  ] + [
    // TODO: combine this with 'gen-'
    {
      name: 'filter-%s%s' % [utils.stacks_to_str(stacks), utils.score_type_post(score_type)],
      score_data_type: score_type,
      stack_filter: stacks,
      sport_filter: 'mlb',
    }
    for stacks in [[5], [4, 3]]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'filter-32' + utils.score_type_post(score_type),
      score_data_type: score_type,
      stack_filter: [3, 2],
      sport_filter: ['nba', 'nhl'],
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'gen-%s' % utils.stacks_to_str(stacks) + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: { stacks: stacks },
      sport_filter: 'mlb,nhl',
    }
    for stacks in [[5], [4]]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'gen-%s' % utils.stacks_to_str(stacks),
      gen_lineups_params: {
        stacks: stacks,
      },
      sport_filter: 'mlb,nhl',
    }
    for stacks in [[4, 3], [3, 3], [3, 2]]
  ] + [
    {
      name: 'player-pctl:%s:%d%s' % [utils.stacks_to_str(stacks), pctl, utils.score_type_post(score_type)],
      score_data_type: score_type,
      gen_lineups_params: {
        stacks: stacks,
        stackable_players_ppg_pctl: pctl,
      },
      sport_filter: 'mlb,nhl',
    }
    for stacks in [[5], [4], [3, 3]]
    for pctl in [30, 40, 50]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'player-pctl-43.%d' % pctl,
      sport_filter: '!nfl',
      gen_lineups_params: {
        stacks: [4, 3],
        stackable_players_ppg_pctl: pctl,
      },
    }
    for pctl in [50, 40, 30]
  ] + [
    {
      name: 'player-pctl-%s.%d' % [utils.stacks_to_str(stacks), pctl],
      sport_filter: '!nfl',
      gen_lineups_params: {
        stacks: stacks,
        stackable_players_ppg_pctl: pctl,
      },
    }
    for pctl in [50, 40, 30]
    for stacks in [[2, 2], [2, 2, 2]]
  ] + [
    {
      name: 'top-games-%s:%d' % [utils.stacks_to_str(stacks), bets],
      sport_filter: 'nhl,nba',
      multi_lineup_stack_top_games: stacks + [65],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2, 3]
    for stacks in [[4, 3], [3, 3]]
  ] + [
    {
      name: 'top-games-21:%d' % bets,
      sport_filter: 'nfl',
      multi_lineup_stack_top_games: [2, 1, 65],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2, 3]
  ] + [
    {
      name: 'top-teams-%s:%d' % [utils.stacks_to_str(stacks), bets],
      sport_filter: 'mlb,nhl',
      multi_lineup_stack_top_teams: [70, stacks],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2, 3]
    for stacks in [[4, 3], [5]]
  ],
}
