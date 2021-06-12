import torch
import torch.nn as nn

from cherry_rl.algorithms.nn.actor_critic import init
from cherry_rl.algorithms.nn.conv_encoders import cnn_forward


class DoomCNN(nn.Module):
    """
    Simplified CNN from the 'sample factory' paper.
    """
    def __init__(self, input_channels):
        super().__init__()
        gain = nn.init.calculate_gain('relu')

        # (input_channels, 128, 72) -> (128, 6, 3); 128 * 6 * 3 = 2304
        self.conv = nn.Sequential(
            init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(64, 128, kernel_size=3, stride=2), gain=gain), nn.ReLU()
        )

    def forward(self, observation):
        return cnn_forward(self.conv, observation)


class DoomFeaturesCNN(DoomCNN):
    """
    Full CNN with features embedding from the 'sample factory' paper.
    """
    def __init__(self, input_channels, n_features):
        super().__init__(input_channels)

        gain = nn.init.calculate_gain('relu')
        self.features_embedding = nn.Sequential(
            init(nn.Linear(n_features, 128), gain=gain), nn.ReLU(),
            init(nn.Linear(128, 128), gain=gain), nn.ReLU(),
        )

    def forward(self, observation):
        img, features = observation['img'], observation['features']
        img_embedding = super().forward(img)
        features_embedding = self.features_embedding(features)
        embedding = torch.cat((img_embedding, features_embedding), dim=-1)
        return embedding
