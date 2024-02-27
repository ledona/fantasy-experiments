from torch import nn
import torch
from fantasy_py import log

_LOGGER = log.get_logger(__name__)


class DeepLineupModel(nn.Module):
    def __init__(self, inventory_size: int, kernel_size=3):
        """
        inventory_size: number of rows in the input matrix,\
            i.e number of player/teams in the slate
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=inventory_size, out_channels=inventory_size, kernel_size=kernel_size
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(inventory_size, inventory_size)

        # self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()

    def forward(self, x):
        """x: is (batch_size, seq_len, input_size)"""
        x = self.conv1(x)  # x is now (batch_size, seq_len, out_channels)
        x = self.pool(x)  # x is now (batch_size, 1, out_channels)

        x = x.squeeze()  # output is (batch_size, out_channels)

        x = self.fc(x)  # x is now (batch_size, out_channels)
        # x = self.sigmoid(x)
        # x = (1 + self.tanh(x)) / 2

        return x


def save(model: DeepLineupModel, filepath: str):
    _LOGGER.info("Saving model to '%s'", filepath)
    torch.save(model, filepath)


def load(filepath: str):
    _LOGGER.info("Loading model from '%s'", filepath)
    return torch.load(filepath)
