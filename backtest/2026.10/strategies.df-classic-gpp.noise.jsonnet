{
  name: 'df-classic-noise',
  params: {
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    max_bets_per_epoch: 1,
    max_bets_per_contest: 1,
  },
  filepath: 'config/strategy/df-classic-gpp.json',
  strategies: [
    {
      name: 'baseline',
    },
  ] + [
    {
      name: 'natural-multi-w/noise-%d:%s:%s' % [bets, noise_diff[0] * 100, noise_diff[1] * 100],
      noise: noise_diff[0],
      diff_pct: noise_diff[1],
      max_bets_per_epoch: bets,
      max_bets_per_contest: bets,
    }
    for bets in [2, 3, 4]
    for noise_diff in [[0.01, 0.25], [0.03, 0.4], [0.05, 0.25]]
  ],
}
