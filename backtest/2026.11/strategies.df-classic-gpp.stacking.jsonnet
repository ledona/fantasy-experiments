// Keep this file close to plain JSON: explicit strategy blocks and simple grid comprehensions; no helper
// functions, object extension or conditional field names. Add a 1-2 line attribution comment to changed blocks.
local utils = import 'utils.libsonnet';
{
  name: 'df-classic-stacking',
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
  // 2026.11: ceiling variants only for nfl/nhl, ceiling lost to predicted in nba/mlb classic (2026.10-2)
  strategies: [
    {
      name: 'baseline%s' % utils.score_type_post(score_type),
      score_data_type: score_type,
      sport_filter: if score_type == 'ceiling' then '!nba,!mlb' else null,
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'player-pctl-22.40:ceiling',
      score_data_type: 'ceiling',
      sport_filter: 'nhl',
      gen_lineups_params: {
        stacks: [2, 2],
        stackable_players_ppg_pctl: 40,
      },
    },
    {
      name: 'top-games-43:1:ceiling',
      sport_filter: 'nhl',
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
    for stacks in [[4], [3, 3], [5], [4, 3]]
  ] + [
    {
      name: 'filter-32' + utils.score_type_post(score_type),
      score_data_type: score_type,
      stack_filter: [3, 2],
      sport_filter: if score_type == 'ceiling' then ['nhl'] else ['nba', 'nhl'],
    }
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'gen-%s' % utils.stacks_to_str(stacks) + utils.score_type_post(score_type),
      score_data_type: score_type,
      gen_lineups_params: { stacks: stacks },
      sport_filter: if score_type == 'ceiling' then 'nhl' else 'mlb,nhl',
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
    for stacks in [[3, 3], [3, 2]]
  ] + [
    {
      name: 'player-pctl:%s:%d%s' % [utils.stacks_to_str(stacks), pctl, utils.score_type_post(score_type)],
      score_data_type: score_type,
      gen_lineups_params: {
        stacks: stacks,
        stackable_players_ppg_pctl: pctl,
      },
      sport_filter: if score_type == 'ceiling' then 'nhl' else 'mlb,nhl',
    }
    for stacks in [[5], [4], [3, 3]]
    for pctl in [30, 40, 50]
    for score_type in ['predicted', 'ceiling']
  ] + [
    {
      name: 'player-pctl-22.%d' % pctl,
      // 2026.11: nba excluded, forced stacks lost in nba (2026.10-2 NBA §4)
      sport_filter: '!nfl,!nba',
      gen_lineups_params: {
        stacks: [2, 2],
        stackable_players_ppg_pctl: pctl,
      },
    }
    for pctl in [50, 40, 30]
  ] + [
    {
      name: 'top-games-%s:%d' % [utils.stacks_to_str(stacks), bets],
      // 2026.11: [4,3] nhl only, forced stacks lost in nba (2026.10-2 NBA §4)
      sport_filter: if stacks == [4, 3] then 'nhl' else 'nhl,nba',
      multi_lineup_stack_top_games: stacks + [65],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2]
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
    // 2026.11: lower min-teams-pctl, 65 left too few eligible games (2026.10-2 NFL 2.1.b); no 3rd value = no cutoff
    {
      name: 'top-games-21:%d:pctl-%s' % [bets, if pctl == null then 'none' else pctl],
      sport_filter: 'nfl',
      multi_lineup_stack_top_games: if pctl == null then [2, 1] else [2, 1, pctl],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2, 3]
    for pctl in [null, 50]
  ] + [
    {
      name: 'top-games-%s:%d:pctl-%s' % [utils.stacks_to_str(stacks), bets, if pctl == null then 'none' else pctl],
      sport_filter: 'nhl',
      multi_lineup_stack_top_games: if pctl == null then stacks else stacks + [pctl],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2]
    for stacks in [[4, 3], [3, 3]]
    for pctl in [null, 50]
  ] + [
    // 2026.11: N up to 5 (2026.10-2 best-of-3 +16.9 nhl / +17.2 mlb for top-teams-5:3)
    {
      name: 'top-teams-%s:%d' % [utils.stacks_to_str(stacks), bets],
      sport_filter: 'mlb,nhl',
      multi_lineup_stack_top_teams: [70, stacks],
      max_bets_per_contest: bets,
      max_bets_per_epoch: bets,
    }
    for bets in [1, 2, 3, 4, 5]
    for stacks in [[4, 3], [5]]
  ] + [
    // 2026.11: top-team pctl sweep for the 5-stack rotation
    {
      name: 'top-teams-5:3:pctl-%d' % pctl,
      sport_filter: 'mlb,nhl',
      multi_lineup_stack_top_teams: [pctl, [5]],
      max_bets_per_contest: 3,
      max_bets_per_epoch: 3,
    }
    for pctl in [60, 80]
  ] + [
    // 2026.11: combined with mlb's best single-lineup filter (2026.10-2 MLB prioritize-P +3.8)
    {
      name: 'top-teams-5:3+prioritize-P',
      sport_filter: 'mlb',
      multi_lineup_stack_top_teams: [70, [5]],
      max_bets_per_contest: 3,
      max_bets_per_epoch: 3,
      gen_lineups_params: {
        require_top_predicted_players: ['P', 2],
      },
    },
  ] + [
    // 2026.11: combined with nhl's best single-lineup filter (2026.10-2 NHL opp-vs-pos.70 +5.3)
    {
      name: 'top-teams-5:3+drop-players-against-top-team-vs-pos.70',
      sport_filter: 'nhl',
      multi_lineup_stack_top_teams: [70, [5]],
      max_bets_per_contest: 3,
      max_bets_per_epoch: 3,
      gen_lineups_params: {
        exclude_player_for_opp_team_vs_pos_pctl: { '': 70 },
      },
    },
  ],
}
