"""
End to end test. Runs the cli for each sport on small excerpts of the real source data
(regression/input) using the real season calendars, and compares the result to the reviewed
output in regression/expected. The last 2 rows of the nfl betiq input are a synthetic copy
of a sportsbook game, used to test source priority.
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
    expected_filepath = os.path.join(REGRESSION_DIR, "expected", filename.replace(".parquet", ".csv"))
    assert os.path.isfile(expected_filepath), f"Unexpected archive filename '{filename}'"
    with open(expected_filepath, "r") as f_:
        assert archive_csv == f_.read()
