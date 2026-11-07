// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
{
  name: 'df-classic-nba',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
    sport_filter: 'nba',
  },
  starting_bankroll: 1000,
  allow_negative_bankroll: true,
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    {
      name: 'baseline',
    },
  ] + [
    {
      name: 'rational-1',
      gen_lineups_params: {
        exclude_below_ppg_pctl: 10,
        exclude_below_games_played: 2,
      },
    },
    {
      name: 'rational-2',
      gen_lineups_params: {
        exclude_below_games_played: 2,
      },
    },
    // 2026.11: sentinel for the pruned nba forced-stack family (2026.10-2 NBA §4)
    {
      name: '3x3.T70.ppg25',
      gen_lineups_params: {
        stacks: [3, 3],
        stackable_teams_predicted_pctl: 70,
        exclude_below_ppg_pctl: 25,
      },
    },
  ] + [
    // 2026.11: per-player cost floor, binds ~55%/80% of 2026.10-2 baseline lineups
    {
      name: 'cost-floor:<%gk' % floor,
      gen_lineups_params: { exclude_for_cost: '<%d' % (floor * 1000) },
    }
    for floor in [3.5, 4]
  ] + [
    // 2026.11: cap the number of expensive players (winner's-curse test, showdown CPT caps helped);
    // binds ~26%/27%/20% of 2026.10-2 baseline lineups
    {
      name: 'max-expensive:%gk-max%d' % [pool[0] / 1000, pool[1]],
      gen_lineups_params: { cost_player_pool: ['>=%d' % pool[0], '<=%d' % pool[1]] },
    }
    for pool in [[8000, 2], [9000, 1], [10000, 0]]
  ],
}
