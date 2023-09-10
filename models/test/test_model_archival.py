import os

from ..model_lib.archive_fantasy_model import archive_model
from ..model_lib.model_lib import retrieve


def test(tmpdir):
    exp_name = "2023.08-test"
    exp_desc = "test experiment description"
    model_name = "LOL-player-DK"
    model_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "LOL-player-DK.dk_performance_score.tpot.model",
    )
    run_tags = {
        "framework": "tpot",
    }
    tracker_settings = {"mlf_tracking_uri": f"file://{tmpdir}"}

    run_id, _ = archive_model(
        model_filepath,
        exp_name,
        experiment_description=exp_desc,
        tracker_settings=tracker_settings,
        run_tags=run_tags,
    )

    models = retrieve(
        run_id=run_id,
        tracker_settings=tracker_settings,
    )

    assert len(models) == 1
    assert models[0].name == model_name
