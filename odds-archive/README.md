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

One row per game, indexed by `date`. Columns: `season, away-team, away-abbr, home-team, home-abbr, away-moneyline, home-moneyline, overunder, over-odds, under-odds, spread`.

- Odds are American style.
- `spread` is the home team's spread: negative means home is favored, 0 is pick'em.
- `over-odds`/`under-odds` are None when not available.

## How it works

1. **Load:** each source (`odds_archive/sources/`) reads its raw files and converts each pair of away/home rows into a game row (`pairing.py`). Seasons come from the sport's season calendar (`seasons.py`). Team abbreviations come from `team_abbrs.json` plus a few season rules (`teams.py`).
2. **Validate:** values outside the sport's valid range are set to null, and games left with no odds are dropped (`schema.py`).
3. **Merge:** sources are combined in priority order. A game (date, away, home) comes from the highest-priority source that has it (`merge.py`).
4. **Report/compare/write:** logs counts per source (games, skipped by reason, values nulled, merge conflicts), optionally compares against a reference archive (`compare.py`), then writes the file.

### Sources

| Name | Sports | Files |
|---|---|---|
| `sportsbook` | all | `*Sheet*.csv`, one season per file |
| `betiq` | mlb, nfl | `{sport}-odds.betiq.tsv` |

### Adding a source

1. Create `odds_archive/sources/{name}.py` with an `OddsSource` subclass (see `sources/base.py`) that sets `NAME`, `SPORTS` and `load(ctx)`. `load` returns a frame in the archive schema, and `pair_games` and `to_archive_frame` usually do most of the work.
2. Register it in `SOURCES` and `SOURCE_PRIORITY` in `sources/__init__.py`.
3. Add unit tests and regression input/expected files.

## Tests

```bash
pytest -q -p no:freezegun
```

The unit tests use small inline data and a stub season calendar. `tests/test_regression.py` runs the CLI on the small real-data excerpts in `tests/regression/input/` and compares the output to the reviewed CSVs in `tests/regression/expected/`.
