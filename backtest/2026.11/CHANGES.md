# 2026.11 backtest changes (from 2026.10-2 results)

## Code (`/fantasy`)
| Area | Change |
|---|---|
| `strategy_classic.py` top-games | fixed unbound `pctl` crash and `None` notes concat; clearer "none eligible" / "already stacked" notes |
| `strategy_classic.py` multi-lineup | `require_top_predicted_players` from `gen_lineups_params` no longer dropped by noise/top-teams/top-games strategies |
| `strategy_showdown.py` | `players_in_pos_lineup_gen` pool too small and `score_cond` `ImputeFailure` now skip the slate, not the epoch |
| `strategy_showdown.py` rank betting | prob categorization uses predicted means (was ~100% lop-sided); confidence = prob of the predicted category; new default prob cutoffs (~p25/p75 totals, ~p75 margin) |
| `create_lineups.py` | solver `user_limit` reported as time/iteration limit, not infeasibility |
| `gen_lineups_params.py` | new `cost_player_pool` (`PlayerCostPool`), e.g. `['>=9000', '<=1']`; CLI `--cost_player_pool` |
| `constraint/opp_team.py` | renamed `no_opp_team`→`opp_team`, `opp_excluded_positions`→`opp_positions`; new `min_opp_players` (bring-back), `max_opp_players: null` = no cap; team items no longer always anchors; tighter big-M (faster solves) |
| `config/strategy/showdown-rank-betting.draftkings.json` | MLB lop-sided plan: `P=0` removed |
| `scripts/bet.py` | accepts strategy `.json` paths; `--strategy_params JSON` to run any backtested strategy |

## Strategies (this directory)
- **Showdown pairing:** file-level `max_bets_per_epoch: null`, so all strategies can bet every slate.
- **Ceiling:** limited to NFL/NHL classic and NFL/NBA showdown; `baseline:ceiling` everywhere plus one ceiling sentinel per pruned sport/contest.
- **Pruned (one sentinel kept per family):** drop-inactive, budget-min/full-budget, CPT floors ≥8k, NBA forced stacks, NBA opp-vs-pos, MLB no-pitcher lineup_gen, expensive NFL RB, `[2,2,2]`/`[4,3]` generic stacks, never-binding duplicates, `vs-G-*` (96 → 4).
- **Fixed definitions:** `nba-rational.1` renamed to list all its constraints; `S-budget` names use `%g`.
- **Rational benchmarks:** MLB rationals add a batting-order stack; NFL `rational.1` adds a bring-back.
- **Derived strategies:** `mlb-from-rational-2:*` and `nfl-from-rational.2:QB-stack+cheap-WRTE`.
- **New grids/combos:**
  - showdown CPT cap × floor (predicted)
  - multi-entry N up to 5, top-teams pctl sweep, NBA position pivots, NFL noise/diff
  - MLB P × tired × team-pctl factorial
  - NHL opp-vs-pos × group × stacks
  - NFL WR,TE × QB × RB budgets
  - top-games min-teams-pctl {none, 50}
  - selection filters + CPT cap
  - best filter + best multi-entry combos
- **New families:**
  - per-player cost floors (`exclude_for_cost`), all sports, classic + showdown
  - max expensive players (`cost_player_pool`), classic
  - NHL `G-stack:2/3`, `G-lockout`
  - `nfl-bring-back`
- **Rank betting:** static ranges from 2026.10-2 pctls (`*-p33-67-90`, `*-p10-50-90`); prob `*-tight`/`*-loose` cutoffs; `min_conf` {0.3, 0.4, 0.5}; rank-betting + CPT <9k.
- **Readability:** each file has a header asking that it stay close to plain JSON; changed blocks carry 2026.11 attribution comments.

### Strategy counts per sport (2026.10-2 in parentheses)
| file | MLB | NBA | NFL | NHL |
|---|---|---|---|---|
| classic misc | 19 (30) | 14 (19) | 26 (30) | 19 (23) |
| classic sport file | 84 (80) | 9 (15) | 74 (54) | 57 (143) |
| classic noise | 9 (7) | 9 (7) | 15 (7) | 9 (7) |
| classic stacking | 34 (49) | 4 (19) | 11 (5) | 58 (50) |
| **classic total** | **146 (166)** | **36 (60)** | **126 (96)** | **143 (223)** |
| showdown | 94 (120) | 83 (97) | 88 (100) | 104 (143) |
| **total** | **240 (286)** | **119 (157)** | **214 (196)** | **247 (366)** |

NFL showdown lineups grow about 4× despite fewer strategies, because the 3-slate cap is gone (about 13 slates per epoch). The other sports' showdown slates average 1.4–3.3 per epoch, so the cap barely mattered there.

## Analysis instructions (`/fantasy/AI-backtest-analysis-instructions.md`)
- **Tightened and de-duplicated;** fixed broken references.
- **$ as the ultimate objective,** weighted by the win-score diagnostics.
- **New rules:**
  - baseline chosen from resolved params
  - classic selection test
  - rational strategies as benchmarks
  - pruning keeps sentinels
  - rank-betting category mix (flag >80% in one category)
  - quote CLI args built from backtest content
- **Optimal params** may be given as `bet.py` commands.

## Not comparable to 2026.10-2
- `rank-betting:prob*`: the method and the default cutoffs changed.
- MLB `rank-betting*`: the lop-sided plan changed.
- Showdown summary totals: no bet cap now.
- `opp_team`/`rational` lineups: anchor fix and bring-back.

## Open / deferred
- per-strategy rank-betting plans and static confidence scale: revisit after 2026.11 results
- stack formulation without combination enumeration (L)
- win-score model refit and a prior NFL season: not done this period
- 2026.10-2 and earlier jsonnet still use `no_opp_team` / `opp_excluded_positions` and won't re-run as-is
