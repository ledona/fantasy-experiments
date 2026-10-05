local utils = import 'utils.libsonnet';
{
  name: 'df-showdown-misc',
  params: {
    service_filter: ['draftkings'],
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    // try to bet 3x per epoch, once per slate
    max_bets_per_epoch: 3,
    max_bets_per_contest: 1,
    max_slates_to_try: null,
  },
  filepath: 'config/strategy/df-showdown-gpp.json',
  strategies: [
    {
      name: 'baseline' + utils.score_type_post(score_type),
      score_data_type: score_type,
    }
    for score_type in ['predicted', 'floor', 'ceiling']
  ] + [
    {
      name: 'rank-betting:%s%s' % [rb_type, utils.score_type_post(score_type)],
      score_data_type: score_type,
      rank_betting: rb_type,
    }
    for score_type in ['predicted', 'ceiling']
    for rb_type in ['static', 'prob']
  ] + [
    {
      name: 'cpt-budget:<%gk' % budget + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: {
        slot_budget: { CPT: '<%d' % (budget * 1000) },
      },
    }
    for budget in [8.5, 9, 9.5, 10, 10.5]
    for score_type in ['predicted', 'ceiling']
  ] + [
    // minimum budget
    {
      name: 'budget-min:46k',
      sport_filter: '!nhl,!nfl',
      gen_lineups_params: { min_budget: 46000 },
    },
  ] + [
    {
      name: 'budget-min:%gk' % budget + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: { min_budget: budget * 1000 },
    }
    for budget in [47.5, 49]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'full-budget:50k' + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: { min_budget: 50000 },
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'mlb-all-off-' + type,
      sport_filter: 'mlb',
      ['players_in_pos_' + type]: ['both', ['P'], '=0'],
    }
    for type in ['filter', 'lineup_gen']
  ] + [
    {
      name: 'mlb-all-off-high-score:>=%s:%s%s' % [min_score, type, utils.score_type_post(score_type)],
      sport_filter: 'mlb',
      score_data_type: score_type,
      ['players_in_pos_' + type]: ['both', ['P'], '=0'],
      score_cond: ['both', '>=%g' % min_score],
    }
    for type in ['filter', 'lineup_gen']
    for min_score in [4.5, 4.75, 4.8, 4.9, 5]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'cpt-budget:<11k',
      sport_filter: '!nba',
      gen_lineups_params: {
        slot_budget: { CPT: '<11000' },
      },
    },
    {
      name: 'mlb-1P',
      sport_filter: 'mlb',
      players_in_pos_filter: [['max', ['P'], '=1'], ['min', ['H'], '<3']],
      score_cond: ['min', '<4'],
    },
    {
      name: 'mlb-1P-CPT',
      sport_filter: 'mlb',
      gen_lineups_params: {
        slot_pos_pools: [['CPT', ['P']]],
      },
      players_in_pos_filter: [['max', ['P'], '=1'], ['min', ['H'], '<3']],
      score_cond: ['min', '<4'],
    },
    {
      name: 'mlb-exclude-P-from-CPT',
      sport_filter: 'mlb',
      gen_lineups_params: {
        slot_pos_exclusions: [['CPT', ['P']]],
      },
    },
    {
      name: 'mlb-opp_hitter_cap',
      sport_filter: 'mlb',
      gen_lineups_params: {
        slot_pos_opp_team_cap: [['CPT', 'P', 2]],
      },
    },
  ] + [
    {
      name: 'mlb-2P-low-score-' + type,
      sport_filter: 'mlb',
      ['players_in_pos_' + type]: ['both', ['P'], '=2'],
      score_cond: ['both', '<4.25'],
    }
    for type in ['filter', 'lineup_gen']
  ] + [
    {
      name: 'mlb-prefer-fav-' + type,
      sport_filter: 'mlb',
      ['players_in_pos_' + type]: [['max', ['P'], '=1'], ['max', ['H'], '>=4']],
      score_cond: ['diff', '>=1'],
    }
    for type in ['filter', 'lineup_gen']
  ] + [
    {
      name: 'nba-prefer-fav.%d' % count,
      sport_filter: ['nba'],
      players_in_pos_lineup_gen: ['max', ['G', 'F', 'C'], '>=%d' % count],
      score_cond: [['max', '>=115'], ['diff', '>=10']],
    }
    for count in [4, 5]
  ] + [
    {
      name: 'nba-prefer-fav.1',
      sport_filter: ['nba'],
      players_in_pos_lineup_gen: ['max', ['G', 'F', 'C'], '>=4'],
      score_cond: [['max', '>115'], ['diff', '>=7']],
      gen_lineups_params: {
        exclude_for_cost: '<2000',
      },
    },
    {
      name: 'nba-prefer-fav.2',
      sport_filter: ['nba'],
      players_in_pos_lineup_gen: ['max', ['G', 'F', 'C'], '>=4'],
      score_cond: [['max', '>115'], ['diff', '>=7']],
    },
    {
      name: 'nba-prefer-fav.3',
      sport_filter: ['nba'],
      players_in_pos_lineup_gen: ['max', ['G', 'F', 'C'], '>=5'],
      score_cond: [['max', '>=115'], ['diff', '>=10']],
      gen_lineups_params: {
        exclude_for_cost: '<2000',
      },
    },
  ] + [
    {
      name: 'nhl-all-offense:%g%s' % [min_score, utils.score_type_post(score_type)],
      score_data_type: score_type,
      sport_filter: ['nhl'],
      players_in_pos_filter: [['both', ['G'], '=0']],
      score_cond: ['both', '>=%g' % min_score],
    }
    for min_score in [2.7, 2.8, 2.9, 3]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'nhl-2G-low-score',
      description: 'place the bet if predicting low scores for both teams and natural lineup has 1 goalies',
      sport_filter: ['nhl'],
      players_in_pos_filter: [['both', ['G'], '=1']],
      score_cond: ['both', '<2.9'],
    },
  ] + [
    {
      name: 'nhl-1G-CPT:' + type,
      sport_filter: ['nhl'],
      gen_lineups_params: {
        slot_pos_pools: [['CPT', ['G']]],
      },
      ['players_in_pos_' + type]: [
        ['max', ['G'], '=1'],
        ['min', ['W', 'C', 'D'], '<3'],
      ],
      score_cond: ['min', '<3'],
    }
    for type in ['filter', 'lineup_gen']
  ] + [
    {
      name: 'nhl-1G:%s:%g%s' % [type, loser_score_max, utils.score_type_post(score_type)],
      description: 'require goalie if opposing team is low scoring',
      sport_filter: ['nhl'],
      score_data_type: score_type,
      ['players_in_pos_' + type]: [
        ['max', ['G'], '=1'],
        ['min', ['W', 'C', 'D'], '<3'],
      ],
      score_cond: ['min', '<%g' % loser_score_max],
    }
    for loser_score_max in [2.8, 2.9, 3, 3.1, 3.2]
    for type in ['filter', 'lineup_gen']
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'nhl-prefer-fav:%s:%g%s' % [type, diff, utils.score_type_post(score_type)],
      description: 'Use lineups favoring heavily favored teams',
      sport_filter: ['nhl'],
      score_data_type: score_type,
      ['players_in_pos_' + type]: [
        ['max', ['W', 'C', 'D'], '>=4'],
      ],
      score_cond: ['diff', '>.6'],
    }
    for type in ['filter', 'lineup_gen']
    for diff in [0.4, 0.5, 0.6, 0.7, 0.8]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'nhl-exclude-G-from-CPT',
      sport_filter: ['nhl'],
      gen_lineups_params: {
        slot_pos_exclusions: [['CPT', ['G']]],
      },
    },
    {
      name: 'nhl-opp_skater_cap',
      sport_filter: ['nhl'],
      gen_lineups_params: {
        slot_pos_opp_team_cap: [['CPT', 'G', 2]],
      },
    },
    {
      name: 'nfl-prefer-fav.3',
      sport_filter: 'nfl',
      players_in_pos_filter: [
        ['max', ['QB'], '=1'],
        ['max', ['WR', 'TE', 'RB'], '>=3'],
      ],
      score_cond: ['diff', '>5'],
    },
    {
      name: 'nfl-2QB-high-score',
      sport_filter: ['nfl'],
      players_in_pos_filter: [['both', ['team.D'], '=0'], ['both', ['QB'], '=2']],
      score_cond: [['max', '>24'], ['min', '<19']],
    },
    {
      name: 'nfl-1QB-high-score',
      sport_filter: ['nfl'],
      players_in_pos_filter: [['max', ['QB'], '=1']],
      score_cond: ['max', '>24'],
    },
    {
      name: 'nfl-1QB-high-score-CPT',
      sport_filter: ['nfl'],
      gen_lineups_params: {
        slot_pos_pools: [['CPT', ['QB']]],
      },
      players_in_pos_filter: [['max', ['QB'], '=1']],
      score_cond: ['max', '>24'],
    },
  ] + [
    {
      name: 'nfl-prefer-fav.1' + utils.score_type_post(score_type),
      sport_filter: 'nfl',
      score_data_type: score_type,
      players_in_pos_filter: [
        ['max', ['QB'], '=1'],
        ['max', ['WR', 'TE', 'RB'], '>=3'],
      ],
      score_cond: ['diff', '>6'],
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'nfl-prefer-fav.2' + utils.score_type_post(score_type),
      sport_filter: 'nfl',
      score_data_type: score_type,
      players_in_pos_filter: [
        ['max', ['QB'], '=1'],
        ['max', ['WR', 'TE', 'RB'], '>=3'],
        ['min', ['QB'], '=0'],
      ],
      score_cond: ['diff', '>6'],
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'nfl-cost-exclude<%dk' % cap,
      sport_filter: 'nfl',
      gen_lineups_params: {
        exclude_for_cost: '<%d' % (cap * 1000),
      },
    }
    for cap in [1, 2]
  ],
}
