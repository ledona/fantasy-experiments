from contextlib import ExitStack

import pytest
from fantasy_py import DFSContestStyle, UnexpectedValueError
from fantasy_py.betting import GeneralPrizePool

from ..lib.modeling.model import model_filenamer

_TEST_CASES = {
    "empty": (UnexpectedValueError, {}),
    "missing-sport": (UnexpectedValueError, {"service": "draftkings"}),
    "missing-style": (
        UnexpectedValueError,
        {"sport": "nhl", "service": "draftkings", "contest_type": GeneralPrizePool},
    ),
    "partial": ("mlb-draftkings", {"sport": "mlb", "service": "draftkings"}),
    "invalid-sport": (UnexpectedValueError, {"sport": "nlb", "service": "draftkings"}),
    "prefix-err-1": (
        UnexpectedValueError,
        {"prefix": "nhl-draftkings-SHOWDOWN-GPP", "sport": "mlb"},
    ),
    "prefix-err-2": (
        UnexpectedValueError,
        {
            "prefix": "nhl-draftkings-SHOWDOWN-GPP",
            "sport": "nhl",
            "service": "draftkings",
            "contest_type": GeneralPrizePool,
        },
    ),
    "complete-kwargs": (
        "nhl-draftkings-SHOWDOWN-GPP-ridge-t:top_log-f:202606",
        {
            "sport": "nhl",
            "service": "draftkings",
            "style": DFSContestStyle.SHOWDOWN,
            "contest_type": GeneralPrizePool,
            "framework": "ridge",
            "target": "top_log",
            "features": "202606",
        },
    ),
    "complete-w-prefix": (
        "nhl-draftkings-SHOWDOWN-GPP-ridge-t:top_log-f:202606",
        {
            "prefix": "nhl-draftkings-SHOWDOWN",
            "contest_type": GeneralPrizePool,
            "framework": "ridge",
            "target": "top_log",
            "features": "202606",
        },
    ),
    "partial-w-prefix": (
        "nhl-draftkings-SHOWDOWN-GPP-ridge",
        {
            "prefix": "nhl-draftkings-SHOWDOWN",
            "contest_type": GeneralPrizePool,
            "framework": "ridge",
        },
    ),
}
"""some basic valid and invalid naming requests"""


@pytest.mark.parametrize("expected_result, kwargs", _TEST_CASES.values(), ids=_TEST_CASES.keys())
def test_backtest_winscore_model_naming(
    expected_result: str | type[UnexpectedValueError], kwargs: dict
):
    """test that backtest win score models naming works"""
    with ExitStack() as exit_stack:
        if not isinstance(expected_result, str):
            exit_stack.enter_context(pytest.raises(expected_result))
        result = model_filenamer(**kwargs)
    if isinstance(expected_result, str):
        assert result == expected_result
