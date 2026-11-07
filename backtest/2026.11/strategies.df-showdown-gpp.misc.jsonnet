// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
local utils = import 'utils.libsonnet';
{
  name: 'df-showdown-misc',
  params: {
    service_filter: 'draftkings',
    max_wager_per_bet: null,
    max_wager_pct_per_epoch: 0.1,
    // 2026.11: bet all slates so every strategy pairs with baseline on all slates (2026.10-2 2.g)
    max_bets_per_epoch: null,
    max_bets_per_contest: 1,
    max_slates_to_try: null,
  },
  missing_inference_cols_threshold_fail: { mlb: 5, nhl: 3 },
  starting_bankroll: 1000,
  allow_negative_bankroll: true,
  filepath: 'config/strategy/df-showdown-gpp.json',
  // 2026.11: ceiling variants limited to nfl/nba, ceiling lost in nhl/mlb showdown (2026.10-2).
  // baseline:ceiling and cpt-budget:<9k:ceiling run for all sports as ceiling sentinels.
  local ceiling_sport_filter = '!nhl,!mlb',
  strategies: [
    {
      name: 'baseline' + utils.score_type_post(score_type),
      score_data_type: score_type,
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'mlb-rational:Pitcher/Hitter Split: starting pitcher CPT -> max of 2 opposing hitters',
      sport_filter: 'mlb',
      gen_lineups_params: {
        slot_pos_opp_team_cap: 'CPT:P:2.DH,C,1B,2B,3B,SS,LF,CF,RF,OF,DH',
        exclude_for_cost: '<=3000',
      },
    },
    {
      // 2026.11: renamed to list all constraints (2026.10-2 NBA 2.1.c)
      name: 'nba-rational.1:CPT >=8.5k, at least 2 top 70th ppg pctl players, no players <2k',
      sport_filter: 'nba',
      gen_lineups_params: {
        slot_budget: { CPT: '>=8500' },
        ppg_pctl_player_pool: [2, 70],
        exclude_for_cost: '<2000',
      },
    },
    {
      name: 'nba-rational.2',
      sport_filter: 'nba',
      gen_lineups_params: {
        ppg_pctl_player_pool: [2, 60],
        exclude_for_cost: '<2000',
      },
    },
    {
      name: 'nba-rational.3',
      sport_filter: 'nba',
      gen_lineups_params: {
        ppg_pctl_player_pool: [2, 40],
        exclude_for_cost: '<2000',
      },
    },
    {
      name: 'nfl-rational:no Defense and opposing QB',
      sport_filter: 'nfl',
      gen_lineups_params: {
        exclude_for_cost: '<2000',
        constraints: [
          ['opp_team', {
            team_position: 'team.D',
            position: 'team.D',
            opp_positions: ['QB'],
          }],
        ],
      },
    },
    {
      name: 'nhl-rational:Goalie/Skater Lockout',
      sport_filter: 'nhl',
      gen_lineups_params: {
        exclude_for_cost: '<=2000',
        slot_pos_opp_team_cap: 'CPT:G:2.W,LW,RW,C',
      },
    },
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
      name: 'rank-betting:prob%s' % utils.score_type_post(score_type),
      score_data_type: score_type,
      rank_betting: 'prob',
      sport_filter: if score_type == 'ceiling' then ceiling_sport_filter else null,
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'rank-betting-static:%s%s' % [sport_range[0], utils.score_type_post(score_type)],
      score_data_type: score_type,
      rank_betting: 'static',
      sport_filter: sport_range[1],
      rank_betting_static_range: sport_range[2],
    }
    for score_type in ['predicted', 'ceiling']
    for sport_range in [
      ['mlb-old', 'mlb', [3, 6, 10]],
      ['mlb-default', 'mlb', null],
      ['nfl-old', 'nfl', [14, 24, 35]],
      ['nfl-default', 'nfl', null],
      ['nba-old', 'nba', [95.0, 110.0, 125.0]],
      ['nba-default', 'nba', null],
      ['nhl-old', 'nhl', [2.0, 3.5, 5.0]],
      ['nhl-default', 'nhl', null],
    ]
    if score_type == 'predicted' || !std.member(['nhl', 'mlb'], sport_range[1])
  ] + [
    // 2026.11: static ranges from 2026.10-2 predicted team score pctls, defaults put 80-100% of slates in one category
    {
      name: 'rank-betting-static:%s' % sport_range[0],
      rank_betting: 'static',
      sport_filter: sport_range[1],
      rank_betting_static_range: sport_range[2],
    }
    for sport_range in [
      ['mlb-p33-67-90', 'mlb', [4.19, 4.69, 5.36]],
      ['mlb-p10-50-90', 'mlb', [3.77, 4.42, 5.36]],
      ['nfl-p33-67-90', 'nfl', [21.0, 24.7, 27.2]],
      ['nfl-p10-50-90', 'nfl', [18.0, 22.8, 27.2]],
      ['nba-p33-67-90', 'nba', [113.0, 118.1, 122.0]],
      ['nba-p10-50-90', 'nba', [108.5, 115.7, 122.0]],
      ['nhl-p33-67-90', 'nhl', [2.76, 3.02, 3.37]],
      ['nhl-p10-50-90', 'nhl', [2.60, 2.88, 3.37]],
    ]
  ] + [
    // 2026.11: prob cutoffs (low total, high total, lop-sided margin) around the new mid-set defaults (2026.10-2 pctls)
    //   tight = totals p33/p67, margin p67; loose = totals p20/p80, margin p85
    {
      name: 'rank-betting:prob:%s' % sport_cutoffs[0],
      rank_betting: 'prob',
      sport_filter: sport_cutoffs[1],
      rank_betting_prob_cutoffs: sport_cutoffs[2],
    }
    for sport_cutoffs in [
      ['mlb-tight', 'mlb', [8.55, 9.17, 0.89]],
      ['mlb-loose', 'mlb', [8.32, 9.57, 1.27]],
      ['nfl-tight', 'nfl', [44.0, 47.2, 6.2]],
      ['nfl-loose', 'nfl', [42.9, 48.2, 8.8]],
      ['nba-tight', 'nba', [227.5, 233.6, 8.1]],
      ['nba-loose', 'nba', [225.5, 237.2, 10.2]],
      ['nhl-tight', 'nhl', [5.71, 5.93, 0.45]],
      ['nhl-loose', 'nhl', [5.58, 6.05, 0.84]],
    ]
  ] + [
    // 2026.11: prob slate selection by confidence (probability the predicted category happens)
    {
      name: 'rank-betting:prob:min-conf-%g' % min_conf,
      rank_betting: 'prob',
      rank_betting_min_conf: min_conf,
    }
    for min_conf in [0.3, 0.4, 0.5]
  ] + [
    // 2026.11: rank-betting on top of the CPT cap, do category plans add anything (2026.10-2 cpt-budget:<9k)
    {
      name: 'rank-betting:prob+cpt-budget:<9k',
      rank_betting: 'prob',
      gen_lineups_params: { slot_budget: { CPT: '<9000' } },
    },
  ] + [
    {
      name: 'rank-betting-static:%s-default+cpt-budget:<9k' % sport,
      rank_betting: 'static',
      sport_filter: sport,
      gen_lineups_params: { slot_budget: { CPT: '<9000' } },
    }
    for sport in ['mlb', 'nfl', 'nba', 'nhl']
  ] + [
    // 2026.11: CPT cap x floor grid, predicted only (2026.10-2 CPT dose-response in nba/nfl/nhl)
    {
      name: 'cpt-budget:<%gk' % cap,
      gen_lineups_params: {
        slot_budget: { CPT: '<%d' % (cap * 1000) },
      },
    }
    for cap in [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
  ] + [
    {
      name: 'cpt-budget-range:%g-%gk' % [floor, cap],
      gen_lineups_params: {
        slot_budget: { CPT: '%d-%d' % [floor * 1000, cap * 1000] },
      },
    }
    for floor in [3, 4, 4.5, 5]
    for cap in [5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5]
    if cap - floor >= 1.5
  ] + [
    // 2026.11: ceiling CPT variants, <9k is the all-sport ceiling sentinel, 3-5k was nfl's best (2026.10-2)
    {
      name: 'cpt-budget:<9k:ceiling',
      score_data_type: 'ceiling',
      gen_lineups_params: { slot_budget: { CPT: '<9000' } },
    },
    {
      name: 'cpt-budget-range:3-5k:ceiling',
      score_data_type: 'ceiling',
      sport_filter: 'nfl',
      gen_lineups_params: { slot_budget: { CPT: '3000-5000' } },
    },
  ] + [
    // 2026.11 sentinels for pruned families: expensive CPT floor, min budget (2026.10-2)
    {
      name: 'cpt-budget-range:9-12k',
      gen_lineups_params: { slot_budget: { CPT: '9000-12000' } },
    },
    {
      name: 'budget-min:49k',
      gen_lineups_params: { min_budget: 49000 },
    },
  ] + [
    {
      name: 'mlb-all-off-' + type,
      sport_filter: 'mlb',
      ['players_in_pos_' + type]: ['both', ['P'], '=0'],
    }
    for type in ['filter', 'lineup_gen']
  ] + [
    {
      name: 'mlb-all-off-high-score:>=%g:%s' % [min_score, type],
      sport_filter: 'mlb',
      ['players_in_pos_' + type]: ['both', ['P'], '=0'],
      score_cond: ['both', '>=%g' % min_score],
    }
    for type in ['filter', 'lineup_gen']
    for min_score in [4.5, 4.75, 4.8, 4.9, 5]
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
    // 2026.11: best selection filters combined with a CPT cap (2026.10-2 MLB/NHL showdown selection results)
    {
      name: 'mlb-1P+cpt-budget:<10.5k',
      sport_filter: 'mlb',
      gen_lineups_params: { slot_budget: { CPT: '<10500' } },
      players_in_pos_filter: [['max', ['P'], '=1'], ['min', ['H'], '<3']],
      score_cond: ['min', '<4'],
    },
    {
      name: 'nhl-1G:filter:3+cpt-budget-range:4.5-6.5k',
      sport_filter: 'nhl',
      gen_lineups_params: { slot_budget: { CPT: '4500-6500' } },
      players_in_pos_filter: [['max', ['G'], '=1'], ['min', ['W', 'C', 'D'], '<3']],
      score_cond: ['min', '<3'],
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
      name: 'nhl-all-offense:%g' % min_score,
      sport_filter: ['nhl'],
      players_in_pos_filter: [['both', ['G'], '=0']],
      score_cond: ['both', '>=%g' % min_score],
    }
    for min_score in [2.7, 2.8, 2.9, 3]
  ] + [
    {
      name: 'nhl-2G-low-score',
      description: 'place the bet if predicting low scores for both teams and natural lineup has 2 goalies',
      sport_filter: ['nhl'],
      players_in_pos_filter: [['both', ['G'], '=2']],
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
      name: 'nhl-1G:%s:%g' % [type, loser_score_max],
      description: 'require goalie if opposing team is low scoring',
      sport_filter: ['nhl'],
      ['players_in_pos_' + type]: [
        ['max', ['G'], '=1'],
        ['min', ['W', 'C', 'D'], '<3'],
      ],
      score_cond: ['min', '<%g' % loser_score_max],
    }
    for loser_score_max in [2.8, 2.9, 3, 3.1, 3.2]
    for type in ['filter', 'lineup_gen']
  ] + [
    {
      name: 'nhl-prefer-fav:%s:%g' % [type, diff],
      description: 'Use lineups favoring heavily favored teams',
      sport_filter: ['nhl'],
      ['players_in_pos_' + type]: [
        ['max', ['W', 'C', 'D'], '>=4'],
      ],
      score_cond: ['diff', '>%s' % diff],
    }
    for type in ['filter', 'lineup_gen']
    for diff in [0.4, 0.5, 0.6, 0.7, 0.8]
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
  ] + [
    // 2026.11: per-player cost floor per sport, binds ~66% (mlb), 37% (nba), 69% (nfl), 76% (nhl)
    // of 2026.10-2 baseline FLEX slots
    {
      name: 'cost-floor:<%gk' % sport_floor[1],
      sport_filter: sport_floor[0],
      gen_lineups_params: { exclude_for_cost: '<%d' % (sport_floor[1] * 1000) },
    }
    for sport_floor in [['mlb', 4], ['nba', 4], ['nfl', 3], ['nhl', 3]]
  ],
}
