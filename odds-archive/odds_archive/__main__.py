"""
Create the odds archive file for a sport by combining odds data sources.
Run from the odds-archive directory: python -m odds_archive --help
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import pandas as pd
from fantasy_py import log

from .compare import Comparison, compare
from .merge import combine
from .schema import SPORTS, drop_no_odds, validate
from .seasons import SeasonCalendar, sport_calendar
from .sources import SOURCE_PRIORITY, SOURCES, SourceContext
from .teams import TeamAbbrs

_LOGGER = log.get_logger(__name__)


@dataclass
class SourceReport:
    games: int
    skipped: dict[str, int]
    nulled: dict[str, int]
    seasons: list[int]


@dataclass
class ArchiveReport:
    sources: dict[str, SourceReport] = field(default_factory=dict)
    conflicts_df: pd.DataFrame | None = None
    """lower priority rows dropped during combine"""

    def log(self):
        for name, report in self.sources.items():
            _LOGGER.info(
                "source '%s': games=%i seasons=%s skipped=%s nulled=%s",
                name,
                report.games,
                report.seasons,
                report.skipped,
                {col: count for col, count in report.nulled.items() if count > 0},
            )
        if self.conflicts_df is not None and len(self.conflicts_df) > 0:
            _LOGGER.warning(
                "%i lower priority rows dropped due to overlap:\n%s",
                len(self.conflicts_df),
                self.conflicts_df,
            )


def build_archive(
    sport: str,
    source_names: list[str],
    input_dir: str,
    calendar: SeasonCalendar,
    teams: TeamAbbrs,
) -> tuple[pd.DataFrame, ArchiveReport]:
    """source_names in priority order, highest first"""
    report = ArchiveReport()
    source_dfs = {}
    for name in source_names:
        source = SOURCES[name]()
        if sport not in source.SPORTS:
            raise ValueError(f"Source '{name}' does not support {sport=}")
        ctx = SourceContext(sport, input_dir, calendar, teams)
        _LOGGER.info("loading source '%s'", name)
        source_df, nulled = validate(source.load(ctx), sport)
        source_df, no_odds_count = drop_no_odds(source_df)
        if no_odds_count > 0:
            ctx.skipped["no valid odds"] += no_odds_count
        source_dfs[name] = source_df
        report.sources[name] = SourceReport(
            len(source_df), dict(ctx.skipped), nulled, sorted(source_df.season.unique().tolist())
        )

    archive_df, report.conflicts_df = combine(source_dfs)
    return archive_df, report


def archive_filename(archive_df: pd.DataFrame, sport: str, ext: str) -> str:
    return f"{sport}.odds-archive.{archive_df.season.min()}-{archive_df.season.max()}.{ext}"


def _write(archive_df: pd.DataFrame, filepath: str, ext: str):
    if ext == "parquet":
        archive_df.to_parquet(filepath)
        return
    if ext == "csv":
        archive_df.to_csv(filepath)
        return
    raise ValueError(f"Unknown output format '{ext}'")


def _write_diffs(comparison: Comparison, diff_dir: str, sport: str):
    os.makedirs(diff_dir, exist_ok=True)
    for name, diff_df in [
        ("only-new", comparison.only_new_df),
        ("only-ref", comparison.only_ref_df),
        ("value-diffs", comparison.diffs_df),
    ]:
        filepath = os.path.join(diff_dir, f"{sport}.{name}.csv")
        diff_df.to_csv(filepath)
        _LOGGER.info("wrote %i rows to '%s'", len(diff_df), filepath)


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(
        prog="odds_archive", description="Create an odds archive file for a sport"
    )
    parser.add_argument("sport", choices=SPORTS)
    parser.add_argument(
        "--sources",
        help="Comma separated source names in priority order (highest first). "
        f"Default is by sport: {SOURCE_PRIORITY}",
    )
    parser.add_argument(
        "--input-dir",
        help="Source data directory. Default is $FANTASY_DATA_ARCHIVE_DIR/odds-archive/{SPORT}",
    )
    parser.add_argument("--out-dir", help="Directory to write the archive file to")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--dry-run", action="store_true", help="Build the archive but don't write it")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output file")
    parser.add_argument("--compare", metavar="REF_FILE", help="Reference archive parquet to compare to")
    parser.add_argument("--diff-dir", help="Write all --compare differences as CSV files to this dir")
    args = parser.parse_args(argv)
    if args.diff_dir is not None and args.compare is None:
        parser.error("--diff-dir requires --compare")
    if not args.dry_run and args.out_dir is None:
        parser.error("--out-dir is required unless --dry-run")
    return args


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    log.configure_logging()

    source_names = args.sources.split(",") if args.sources else SOURCE_PRIORITY[args.sport]
    input_dir = args.input_dir or os.path.join(
        os.environ["FANTASY_DATA_ARCHIVE_DIR"], "odds-archive", args.sport
    )
    archive_df, report = build_archive(
        args.sport,
        source_names,
        input_dir,
        sport_calendar(args.sport),
        TeamAbbrs.from_file(args.sport),
    )
    report.log()
    _LOGGER.info("archive has %i games", len(archive_df))

    if args.compare is not None:
        comparison = compare(archive_df, pd.read_parquet(args.compare))
        comparison.log()
        if args.diff_dir is not None:
            _write_diffs(comparison, args.diff_dir, args.sport)

    if args.dry_run:
        _LOGGER.info("dry run, not writing archive")
        return
    filepath = os.path.join(args.out_dir, archive_filename(archive_df, args.sport, args.format))
    if os.path.exists(filepath) and not args.overwrite:
        sys.exit(f"'{filepath}' already exists, use --overwrite to replace it")
    _write(archive_df, filepath, args.format)
    _LOGGER.success("archive written to '%s'", filepath)


if __name__ == "__main__":
    main()
