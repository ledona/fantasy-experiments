"""
Runs the cli per sport on real data excerpts (regression/input) and compares to the reviewed
output (regression/expected). The last 2 nfl betiq rows are synthetic, a copy of a game also in
nflverse and sportsbook, to test source priority.
"""

import os

import pandas as pd
import pytest
from odds_archive.__main__ import main
from odds_archive.schema import SPORTS

REGRESSION_DIR = os.path.join(os.path.dirname(__file__), "regression")


def run_regression(sport: str, out_dir: str) -> tuple[str, str]:
    """returns (archive filename, archive as csv text)"""
    main([sport, "--input-dir", os.path.join(REGRESSION_DIR, "input", sport), "--out-dir", out_dir])
    [filename] = os.listdir(out_dir)
    return filename, pd.read_parquet(os.path.join(out_dir, filename)).to_csv()


@pytest.mark.parametrize("sport", SPORTS)
def test_regression(sport, tmp_path):
    filename, archive_csv = run_regression(sport, str(tmp_path))
    expected_filepath = os.path.join(
        REGRESSION_DIR, "expected", filename.replace(".parquet", ".csv")
    )
    assert os.path.isfile(expected_filepath), f"Unexpected archive filename '{filename}'"
    with open(expected_filepath, "r") as f_:
        assert archive_csv == f_.read()
