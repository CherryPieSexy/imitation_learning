import torch.nn as nn

from cherry_rl.algorithms.nn.actor_critic import init
from cherry_rl.algorithms.nn.conv_encoders import cnn_forward


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        gain = nn.init.calculate_gain('relu')

        self._block = nn.Sequential(
            nn.ReLU(),
            init(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                weight_init=nn.init.xavier_uniform_, gain=gain
            ),
            nn.ReLU(),
            init(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                weight_init=nn.init.xavier_uniform_, gain=gain
            )
        )

    def forward(self, block_input):
        block_output = block_input + self._block(block_input)
        return block_output


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels=4, blocks_channels=None):
        super().__init__()
        gain = nn.init.calculate_gain('relu')

        self.conv = []
        if blocks_channels is None:
            blocks_channels = [16, 32, 32, 32]
        for out_channels in blocks_channels:
            self.conv.append(self._init_block(input_channels, out_channels, gain))
            input_channels = out_channels

        # cnn: 97x72xC -> 6x5xC' = 30 * C'
        self.conv = nn.Sequential(*self.conv)

    @staticmethod
    def _init_block(in_channels, out_channels, gain):
        block = nn.Sequential(
            init(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                weight_init=nn.init.xavier_uniform_, gain=gain
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            _ResidualBlock(out_channels),
            _ResidualBlock(out_channels)
        )
        return block

    def forward(self, observation):
        return cnn_forward(self.conv, observation)
