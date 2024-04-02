import json
import os
import platform
from typing import Literal, TypedDict, cast

import torch
from fantasy_py import log
from torch import nn

from .loader import DeepLineupDataset

_LOGGER = log.get_logger(__name__)


class DeepLineupModel(nn.Module):
    player_count: int
    """the number of players included in each slate (including padding)"""

    def __init__(self, player_count: int, player_features: int, hidden_size=128):
        """
        player_count: the number of players available in a slate (including padding)
        slate_features: the number of features per players
        """
        super().__init__()

        _LOGGER.info(
            "Initializing model for player-count=%i player-features=%i => input-size=%i hidden-size=%i output-size=%i",
            player_count,
            player_features,
            player_features * player_count,
            hidden_size,
            player_count,
        )

        if not (player_count <= hidden_size <= player_features * player_count):
            _LOGGER.warning(
                "Hidden layer size is not between model input and output sizes. "
                "Recommended that the hidden size be between %i and %i.",
                player_count,
                player_features * player_count,
            )

        self.player_count = player_count
        self._hidden_size = hidden_size
        self._player_features = player_features

        self.stack = nn.Sequential(
            nn.Linear(player_features * player_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, player_count),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        """model input can be a single sample or a batch, the shape of the returned tensor will
        match the type of input (i.e. single sample input will be a scalar)"""
        assert x.ndim in (2, 3) and list(x.size())[-2:] == [
            self.player_count,
            self._player_features,
        ], (
            "Expecting x to be a single or batch input of 2 or 3 dimensions, "
            "where the later 2 dimensions are of size (num-players, num-features)"
        )
        x_reshaped = x.view(x.size(0), -1) if x.ndim == 3 else x.reshape(1, -1)
        y = self.stack(x_reshaped)

        # if input was a single sample then return a single value else return batch
        return y if x.ndim == 3 else y[0]


class ModelFileData(TypedDict):
    model: DeepLineupModel
    last_epoch: int
    addl_info: dict
    batch_size: int


def save(
    base_filepath: str,
    model: DeepLineupModel,
    epoch: int,
    batch_size,
    model_type: Literal["checkpoint", "final"],
    dataset: DeepLineupDataset,
    performance: dict | None = None,
    **addl_info,
):
    """
    save the model file and model metadata

    base_filepath: path + base filename (without extension) for the artifact file
    model_type: if best-score then a model metadata is stored as json in
        a .model
    """
    artifact_filepath = base_filepath + ".pt"
    _LOGGER.info("Saving %s model artifact to '%s'", model_type, artifact_filepath)

    file_data: ModelFileData = {
        "model": model,
        "last_epoch": epoch,
        "addl_info": addl_info,
        "batch_size": batch_size,
    }
    torch.save(file_data, artifact_filepath)
    if model_type != "final":
        return

    if performance is None:
        _LOGGER.warning("No performance provided for %s", base_filepath)

    meta_filename = base_filepath + ".model"
    _LOGGER.info("Saving %s model meta data to '%s'", model_type, meta_filename)

    name_and_dt = base_filepath.rsplit(os.sep, 1)[1]
    name, dt_trained = name_and_dt.rsplit(".", 1)
    meta_dict = {
        "name": name,
        "sport": dataset.samples_meta["sport"],
        "dt_trained": dt_trained,
        "type": "deep-lineup",
        "parameters": {
            "batch_size": batch_size,
            "epochs": epoch,
            "samples": len(dataset),
            "hidden_size": model._hidden_size,
        },
        "trained_parameters": {
            "model_path": base_filepath,
        },
        "meta_extra": {"performance": performance, "trained_on_uname": platform.uname()._asdict()},
        "training_data_def": {"input_cols": dataset.input_cols},
    }

    with open(meta_filename, "w") as f_:
        json.dump(meta_dict, f_, indent="\t")


def load(filepath: str):
    _LOGGER.info("Loading model from '%s'", filepath)
    return cast(ModelFileData, torch.load(filepath))
