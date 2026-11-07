// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
local utils = import 'utils.libsonnet';
{
  name: 'df-classic-noise',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
  },
  missing_inference_cols_threshold_fail: { mlb: 1 },
  starting_bankroll: 1000,
  allow_negative_bankroll: true,
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    {
      name: 'baseline',
    },
  ] + [
    // 2026.11: noise 0.01 dropped (mostly duplicate lineups), N up to 5 (2026.10-2)
    {
      name: 'natural-multi-w/noise-%d:%g:%g' % [bets, noise_diff[0] * 100, noise_diff[1] * 100],
      noise: noise_diff[0],
      diff_pct: noise_diff[1],
      max_bets_per_epoch: bets,
      max_bets_per_contest: bets,
    }
    for bets in [2, 3, 4, 5]
    for noise_diff in [[0.05, 0.25], [0.07, 0.25]]
  ] + [
    // 2026.11: higher noise and diff for nfl, noise 0.07 best-of-3 +21.0 (2026.10-2 NFL §4)
    {
      name: 'natural-multi-w/noise-3:%g:%g' % [noise_diff[0] * 100, noise_diff[1] * 100],
      sport_filter: 'nfl',
      noise: noise_diff[0],
      diff_pct: noise_diff[1],
      max_bets_per_epoch: 3,
      max_bets_per_contest: 3,
    }
    for noise_diff in [[0.07, 0.4], [0.10, 0.25], [0.10, 0.4], [0.15, 0.25], [0.15, 0.4]]
  ] + [
    // 2026.11: nfl noise combined with its best single-lineup filter (2026.10-2 cheap-WRTE +7.1)
    {
      name: 'natural-multi-w/noise-3:7:25+cheap-WRTE.gen<20k',
      sport_filter: 'nfl',
      noise: 0.07,
      diff_pct: 0.25,
      max_bets_per_epoch: 3,
      max_bets_per_contest: 3,
      gen_lineups_params: {
        group_cost_budget: { 'WR,TE': 20000 },
      },
    },
  ],
}
