from typing import cast

import torch
from fantasy_py import log
from torch import nn

_LOGGER = log.get_logger(__name__)


class DeepLineupModel(nn.Module):
    _player_count: int
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

        self._player_count = player_count
        self._hidden_size = hidden_size
        self._player_features = player_features

        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(player_features * player_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, player_count),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        y = self.stack(x)
        return y


def save(
    filepath: str,
    model: DeepLineupModel,
    epoch: int,
    rewards: list[float],
    **addl_info,
):
    _LOGGER.info("Saving model to '%s'", filepath)
    torch.save(
        {"model": model, "epoch": epoch, "rewards": rewards, "addl_info": addl_info}, filepath
    )


def load(filepath: str):
    _LOGGER.info("Loading model from '%s'", filepath)
    return cast(dict, torch.load(filepath))
