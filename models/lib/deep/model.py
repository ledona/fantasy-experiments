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

        self._player_count = player_count

        self.stack = nn.Sequential(
            nn.Linear(player_features * player_count, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, player_count),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, seq_len, _ = cast(tuple[int, ...], x.size())
        assert seq_len == self._player_count
        y = self.stack(x.view(batch_size, -1))

        # y = x.view(batch_size, -1)
        # y = self.fc1(y)
        # y = self.layer_norm1(y)
        # y = self.relu(y)

        # y = self.fc2(y)
        # y = self.layer_norm2(y)
        # y = self.sigmoid(y)

        return y


def save(model: DeepLineupModel, filepath: str):
    _LOGGER.info("Saving model to '%s'", filepath)
    torch.save(model, filepath)


def load(filepath: str):
    _LOGGER.info("Loading model from '%s'", filepath)
    return torch.load(filepath)
