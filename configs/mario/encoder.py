import torch.nn as nn

from cherry_rl.algorithms.nn.conv_encoders import init, cnn_forward


class Encoder(nn.Module):
    def __init__(self, layer_norm=False):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        self.conv = nn.Sequential(
            init(nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), gain=gain), nn.ReLU(),
        )
        self.linear = nn.Sequential(
            init(nn.Linear(32 * 6 * 6, 512), gain=gain), nn.ReLU()
        )
        self.layer_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(512)

    def forward(self, observation):
        cnn_out = cnn_forward(self.conv, observation)
        linear_out = self.linear(cnn_out)
        if self.layer_norm is not None:
            linear_out = self.layer_norm(linear_out)
        return linear_out
