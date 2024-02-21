"""test deep learning lineup modeling functions"""

import os

import pandas as pd
import pytest
from fantasy_py.lineup.services.fantasy_service import SportConstraints

from ..deep.loss import DeepLineupLoss

_CONSTRAINTS = {
    "constraints": {"A": 2, "B": 1, "C": 1, ("A", "B"): 1},
    "budget": 1200,
    "knapsack_viability_testers": {
        "multiteam": {"min_teams": 3},
        "max_team_players": {"max_players": 2},
    },
}
"""constraints for loss testing"""
_TOP_LINEUP = [20, 14, 23, 6, 7]
"""the best target/lineup for the loss function testing"""
_TOP_SCORE = 50
"""expected score of the top lineup"""


@pytest.fixture(name="sample_base_df", scope="module")
def _sample_base_df():
    df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_sample_base.csv")
    return pd.read_csv(df_path)


def _pred_from_lineup(lineup, sample_base_df: pd.DataFrame):
    """return the predicted lineup dataframe"""
    raise NotImplementedError()


@pytest.mark.parametrize(
    "expected_loss, lineup",
    [
        (0, _TOP_LINEUP),
        (_TOP_SCORE - 32.2, [1, 2, 8, 9, 15]),
        (_TOP_SCORE - 37, [1, 2, 3, 9, 15]),
        (_TOP_SCORE + 5, []),
        (_TOP_SCORE + 4, [1]),
        (_TOP_SCORE + 3, [1, 6, 2]),
        (_TOP_SCORE + 2, [1, 2, 3]),
        (_TOP_SCORE + 1, [1, 2, 3, 8]),
        (_TOP_SCORE + 5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (_TOP_SCORE + 1.6, [18, 19, 15, 6, 7]),
        (_TOP_SCORE + 1, [1, 2, 3, 9, 10]),
        (_TOP_SCORE + 1, [6, 7, 8, 9, 10]),
        (_TOP_SCORE + 1, [1, 2, 8, 9, 16]),
        (_TOP_SCORE + 2, [1, 2, 5, 6, 16]),
    ],
    ids=[
        "top lineup",
        "valid lineup 1",
        "valid lineup 2",
        "empty lineup",
        "missing 4",
        "missing 3",
        "missing 2",
        "missing 1",
        "too many players",
        "over budget",
        "failed 1 viability",
        "failed 2 viability",
        "missing 1 position",
        "missing 2 position",
    ],
)
def test_loss(expected_loss: float, lineup: list[int], sample_base_df: pd.DataFrame):
    """
    lineup: list of player IDs
    """
    dll = DeepLineupLoss([], [], SportConstraints.from_json_dict(_CONSTRAINTS))
    pred = _pred_from_lineup(lineup, sample_base_df)
    loss = dll.calc_loss(pred, _TOP_SCORE)
    assert loss == expected_loss
