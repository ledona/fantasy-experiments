import os

from .archive_fantasy_model import archive_model


def test():
    exp_name = "2023.08-test"
    exp_desc = "test experiment description"
    model_filepath = os.path.join(
        os.path.sep,
        "fantasy-experiments",
        "models",
        "2023.03",
        "LOL-player-DK.dk_performance_score.tpot.model",
    )
    run_tags = {
        "framework": "tpot",
    }
    tracker_settings = {"mlf_tracking_uri": "http://localhost:5000"}

    archive_model(
        model_filepath,
        exp_name,
        experiment_description=exp_desc,
        tracker_settings=tracker_settings,
        run_tags=run_tags,
    )
