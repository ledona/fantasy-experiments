"""test deep learning lineup modeling functions"""

import os

import torch
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
    df = pd.read_csv(df_path)
    df["in-lineup"] = df.player_id.map(lambda pid: pid in _TOP_LINEUP)
    return df


@pytest.fixture(name="top_score", scope="module")
def _top_score(sample_base_df: pd.DataFrame):
    return sample_base_df.query("`in-lineup`")["fpts-historic"].sum()


@pytest.mark.parametrize(
    "loss_delta, lineup",
    [
        (5, []),
        (4, [1]),
        (1, [1, 2, 3, 8]),
        (5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (1.6, [18, 19, 15, 6, 7]),
        (1, [1, 2, 3, 9, 10]),
        (1, [6, 7, 8, 9, 10]),
        (1, [1, 2, 8, 9, 16]),
        (2, [1, 2, 5, 6, 16]),
        (None, [1, 2, 8, 9, 15]),
        (None, [1, 2, 3, 9, 15]),
        (None, _TOP_LINEUP),
    ],
    ids=[
        "empty lineup",
        "missing 4",
        "missing 1",
        "too many players",
        "over budget",
        "failed 1 viability",
        "failed 2 viability",
        "missing 1 position",
        "missing 2 position",
        "valid lineup 1",
        "valid lineup 2",
        "top lineup",
    ],
)
def test_loss(
    loss_delta: None | float, lineup: list[int], sample_base_df: pd.DataFrame, top_score: float
):
    """
    loss_delta: if None then expected loss is top_score - lineup score, if not None then\
        expected loss is top_score + this value
    lineup: list of player IDs
    """
    expected_loss = (
        (top_score + loss_delta)
        if loss_delta is not None
        else top_score - sample_base_df.query("player_id in @lineup")["fpts-historic"].sum()
    )

    dll = DeepLineupLoss([], [], SportConstraints.from_json_dict(_CONSTRAINTS))
    pred = sample_base_df.player_id.map(lambda pid: pid in lineup)
    loss = dll.calc_loss(pred, sample_base_df)
    assert loss == expected_loss
