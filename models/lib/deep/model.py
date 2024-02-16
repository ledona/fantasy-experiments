from torch import nn
import torch
from fantasy_py import log

_LOGGER = log.get_logger(__name__)


class DeepLineupModel(nn.Module):
    def __init__(self, in_channels: int, lineup_slot_count: int):
        """
        in_channels: number of rows in the input matrix
        lineup_slot_count: number of values in the output
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, out_channels=lineup_slot_count, kernel_size=3
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        # x is (batch_size, seq_len, input_size)
        x = self.conv1(x)
        # x is now (batch_size, seq_len, out_channels)
        x = self.pool(x)
        # x is now (batch_size, 1, out_channels)
        return x.squeeze()  # output is (batch_size, out_channels)


def save(model: DeepLineupModel, filepath: str):
    _LOGGER.info("Saving model to '%s'", filepath)
    torch.save(model, filepath)


def load(filepath: str):
    _LOGGER.info("Loading model from '%s'", filepath)
    return torch.load(filepath)
