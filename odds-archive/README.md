# odds_archive

Builds the per-sport historic odds archive (`{sport}.odds-archive.{min_season}-{max_season}.parquet`) from raw odds data files. The archive is read by `_OddsArchive` in `fantasy_py/sport/extra_stats/odds.py` from `$FANTASY_DATA_ARCHIVE_DIR/historic/{sport}/`.

Replaces `sportbook-odds-aggragation.ipynb`.

## Usage

Run from this directory:

```bash
python -m odds_archive SPORT [options]
```

| Option | Description |
|---|---|
| `SPORT` | `mlb`, `nba`, `nfl` or `nhl` |
| `--sources a,b` | sources to use, in priority order (default: per-sport list in `sources/__init__.py`) |
| `--input-dir D` | raw data dir (default: `$FANTASY_DATA_ARCHIVE_DIR/odds-archive/{sport}`) |
| `--out-dir D` | where to write the archive (required unless `--dry-run`) |
| `--format parquet\|csv` | output format (default: parquet) |
| `--dry-run` | build and report, don't write |
| `--overwrite` | allow replacing an existing archive file |
| `--compare REF_FILE` | diff the result against a reference archive |
| `--diff-dir D` | write `--compare` diffs to CSV files |

Example:

```bash
python -m odds_archive nfl --out-dir /tmp/odds --compare $FANTASY_DATA_ARCHIVE_DIR/historic/nfl/nfl.odds-archive.2009-2022.parquet
```

## Output

One row per game, indexed by `date`. Columns: `season, away-team, away-abbr, home-team, home-abbr, away-moneyline, home-moneyline, overunder, over-odds, under-odds, spread, away-spread-odds, home-spread-odds`.

- Odds are American style.
- `spread` is the home team's spread: negative means home is favored, 0 is pick'em. For mlb/nhl it is the ±1.5 run/puck line. Sportsbook takes it from the mlb run line (2014+) or nhl puck line (2014-15+) when the file has one, otherwise derives it from the moneyline favorite. betiq has it directly.
- `over-odds`/`under-odds`/`away-spread-odds`/`home-spread-odds` are None when not available, and are nulled when their line (`overunder`/`spread`) is null. Spread odds come from nflverse and the sportsbook mlb run line / nhl puck line.
- A few known bad source games are excluded (`_EXCLUDED_GAMES` in `sources/sportsbook.py`).
- `away-team`/`home-team` are names in one style for all sources (e.g. `Kansas City`, `LA Rams`, `NY Yankees`), from `team_names.json`.

## How it works

1. **Load:** each source (`odds_archive/sources/`) reads its raw files and converts them to game rows. Sources with one row per team (sportsbook, betiq) pair away/home rows with `pairing.py`. Seasons come from the sport's season calendar (`seasons.py`).
2. **Team abbrs** (`teams.py`): source team names are mapped with `team_abbrs.json` and a few season rules, then a few source typos are fixed, then the sport's fantasy_py `team_abbr_sub` is applied (e.g. JAX→JAC).
3. **Validate:** values outside the sport's valid range are set to null, and games left with no odds are dropped (`schema.py`).
4. **Merge:** sources are combined in priority order. A game (away, home, date) comes from the highest-priority source that has it. For nfl, dates within 3 days count as the same game, to catch source date errors (`merge.py`).
5. **Team names:** `away-team`/`home-team` are set from the abbrs using `team_names.json`. Its keys are the valid abbrs, and any other abbr is an error.
6. **Report/compare/write:** logs counts per source (games, skipped by reason, values nulled, merge conflicts), optionally compares against a reference archive (`compare.py`), then writes the file.

### Sources

| Name | Sports | Files |
|---|---|---|
| [`sportsbook`](https://www.sportsbookreviewsonline.com/scoresoddsarchives/) | all | `*Sheet*.csv`, one season per file, two rows per game |
| [`betiq`](https://betiq.teamrankings.com/) | mlb, nfl | `{sport}-odds.betiq.tsv`, two rows per game |
| [`nflverse`](https://github.com/nflverse) | nfl | `nflverse-games.csv`, one row per game, seasons 2009-2025 used |

Default priority (highest first): nfl `nflverse, sportsbook, betiq`; mlb `sportsbook, betiq`; nba/nhl `sportsbook`.

### Adding a source

1. Create `odds_archive/sources/{name}.py` with an `OddsSource` subclass (see `sources/base.py`) that sets `NAME`, `SPORTS` and `load(ctx)`. `load` returns a frame in the archive schema built with `to_archive_frame`. Use `pair_games` for two-rows-per-game data, and `ctx.teams.abbr()` for team abbrs.
2. Register it in `SOURCES` and `SOURCE_PRIORITY` in `sources/__init__.py`.
3. If the source uses new team names or abbrs, add them to `team_names.json` or map them in `team_abbrs.json`.
4. Add unit tests and regression input/expected files.

## Tests

```bash
pytest -q -p no:freezegun
```

The unit tests use small inline data and a stub season calendar. `tests/test_regression.py` runs the CLI on the small real-data excerpts in `tests/regression/input/` and compares the output to the reviewed CSVs in `tests/regression/expected/`.
