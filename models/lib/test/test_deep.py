"""test deep learning lineup modeling functions"""

import os

import pandas as pd
import pytest
from fantasy_py.lineup.services.fantasy_service import SportConstraints

from ..deep.loss import DeepLineupLoss

_CONSTRAINTS = {
    "constraints": {"A": 2, "B": 1, "C": 1, ("A", "B"): 1},
    "budget": 1600,
}
"""constraints for loss testing"""
_TESTERS = [
    ("multiteam", {"min_teams": 2}),
    ("max_team_players", {"max_players": 2}),
]
"""viability testers for loss testing"""

_TOP_LINEUP = [20, 13, 14, 6, 7]
"""the best target/lineup for the loss function testing"""


@pytest.fixture(name="sample_base_df", scope="module")
def _sample_base_df():
    df_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_sample_base.csv")
    df = pd.read_csv(df_path)
    df["in-lineup"] = df.player_id.map(lambda pid: pid in _TOP_LINEUP)
    return df


@pytest.mark.parametrize(
    "expected_score, lineup, include_testers",
    [
        (-5, [], False),
        (-4, [1], False),
        (-1, [1, 2, 3, 8], False),
        (-5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], False),
        (-0.2, [18, 19, 15, 11, 12], False),
        (-1, [6, 7, 8, 4, 5], True),
        (-2, [1, 2, 3, 4, 5], True),
        (-0.5, [1, 2, 8, 9, 16], False),  # 'C' is missing
        (-1, [1, 2, 6, 7, 16], False),  # missing 'B' and 'C'
        (None, [1, 2, 8, 9, 15], False),
        (None, [1, 2, 3, 9, 15], False),
        (None, _TOP_LINEUP, False),
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
    expected_score: None | float,
    lineup: list[int],
    sample_base_df: pd.DataFrame,
    include_testers: bool,
):
    """
    expected_score: if None then expected score is just the score of the lineup
    lineup: list of player IDs
    """
    constraints = {**_CONSTRAINTS, "testers": _TESTERS} if include_testers else _CONSTRAINTS
    dll = DeepLineupLoss(
        list(sample_base_df.columns), SportConstraints.from_json_dict(constraints), 100
    )
    pred = list(sample_base_df.player_id.map(lambda pid: pid in lineup))
    score = dll.calc_score(pred, sample_base_df)

    final_expected_score = (
        expected_score or sample_base_df.query("player_id in @lineup")["fpts-historic"].sum()
    )

    assert score == final_expected_score
