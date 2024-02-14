from torch import nn


class DeepLineupModel(nn.Module):
    def __init__(self, lineup_slot_count: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=lineup_slot_count, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        # x is (batch_size, seq_len, input_size)
        x = self.conv1(x)
        # x is now (batch_size, seq_len, out_channels)
        x = self.pool(x)
        # x is now (batch_size, 1, out_channels)
        return x.squeeze()  # output is (batch_size, out_channels)
