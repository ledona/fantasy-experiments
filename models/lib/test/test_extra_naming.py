import pytest

from ..pt_model.train_test import _infer_extra_stat_name_type


@pytest.mark.parametrize(
    "feature, expected_name, expected_type",
    [
        ("extra:venue_The Palace", "venue", "current_extra"),
        ("extra:name", "name", "current_extra"),
        ("extra:name:opp-team", "name", "current_opp_team_extra"),
        ("extra:name:player-team", "name", "current_extra"),
        ("extra:name:recent-2", "name", "hist_extra"),
        ("extra:name:recent-mean", "name", "hist_extra"),
        ("extra:name:std-mean", "name", "hist_extra"),
        ("extra:name:opp-team:recent-mean", "name", "hist_opp_team_extra"),
        ("extra:name:opp-team:recent-2", "name", "hist_opp_team_extra"),
        ("extra:name:player-team:std-mean", "name", "hist_extra"),
    ],
)
def test_extra_stat_name_parse(feature, expected_name, expected_type):
    """
    ensure that given an extra stat feature name, the correct stat name
    and extra stat type can be inferred
    """
    x_name, x_type = _infer_extra_stat_name_type(feature.split(":"))
    assert x_name == expected_name
    assert x_type == expected_type
