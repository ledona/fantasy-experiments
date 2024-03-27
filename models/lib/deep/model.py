import torch
from fantasy_py import log
from torch import nn

_LOGGER = log.get_logger(__name__)


class DeepLineupModel(nn.Module):
    _player_count: int
    """the number of players included in each slate (including padding)"""

    def __init__(self, player_count: int, player_features: int, hidden_size=8):
        """
        player_count: the number of players available in a slate (including padding)
        slate_features: the number of features per players
        """
        super().__init__()

        self._player_count = player_count

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


def save(model: DeepLineupModel, filepath: str):
    _LOGGER.info("Saving model to '%s'", filepath)
    torch.save(model, filepath)


def load(filepath: str):
    _LOGGER.info("Loading model from '%s'", filepath)
    return torch.load(filepath)
