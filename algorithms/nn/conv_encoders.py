import torch
import torch.nn as nn

from algorithms.nn.actor_critic import init, activations_dict


def cnn_forward(cnn, observation):
    if type(observation) is dict:
        img = observation.pop('img')
    else:
        img = observation

    with_time = False
    if img.dim() == 5:
        with_time = True
        obs_size = img.size()
        (time, batch), chw = obs_size[:2], obs_size[2:]
        img = img.view(time * batch, *chw)

    conv_features = cnn(img)
    conv_features = conv_features.view(conv_features.size(0), -1)

    if with_time:
        # noinspection PyUnboundLocalVariable
        conv_features = conv_features.view(time, batch, -1)

    if type(observation) is dict:
        observation['features'] = conv_features
        return observation
    else:
        return conv_features


class ConvEncoder(nn.Module):
    """
    Simple 3-layer CNN with ReLU activation for images of size (input_channels, 42, 42).
    """
    def __init__(self, input_channels=4, activation_str='relu'):
        super().__init__()

        gain = nn.init.calculate_gain(activation_str)
        activation = activations_dict[activation_str]
        # (input_channels, 42, 42) -> (32, 4, 4)
        self.conv = nn.Sequential(
            init(nn.Conv2d(input_channels, 32, kernel_size=3, stride=2), gain=gain), activation(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2), gain=gain), activation(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2), gain=gain), activation()
        )

    def forward(self, observation):
        return cnn_forward(self.conv, observation)


class DeepConvEncoder(nn.Module):
    """
    "Deep" 6-layer CNN with ReLU activation for images of size (input_channels, 96, 96).
    """
    def __init__(self, input_channels=4, activation_str='relu'):
        super().__init__()
        gain = nn.init.calculate_gain(activation_str)
        activation = activations_dict[activation_str]
        # (input_channels, 96, 96) -> (256, 1, 1)
        self.conv = nn.Sequential(
            init(nn.Conv2d(input_channels, 8, kernel_size=4, stride=2), gain=gain), activation(),
            init(nn.Conv2d(8, 16, kernel_size=3, stride=2), gain=gain), activation(),
            init(nn.Conv2d(16, 32, kernel_size=3, stride=2), gain=gain), activation(),
            init(nn.Conv2d(32, 64, kernel_size=3, stride=2), gain=gain), activation(),
            init(nn.Conv2d(64, 128, kernel_size=3, stride=1), gain=gain), activation(),
            init(nn.Conv2d(128, 256, kernel_size=3, stride=1), gain=gain), activation()
        )

    def forward(self, observation):
        return cnn_forward(self.conv, observation)


class FeaturesConvEncoder(nn.Module):
    """
    Applies CNN to image and one Linear layer to rest features.
    """
    def __init__(self, make_cnn, feature_size, embedding_size):
        super().__init__()
        self.cnn = make_cnn()
        self.features_embedding = nn.Linear(feature_size, embedding_size)

    def forward(self, observation):
        img = observation.pop('img')
        conv_features = self.cnn(img)
        obs_features = torch.cat([value for _, value in observation.items()], dim=-1)
        features_emb = self.features_embedding(obs_features)
        result = torch.cat([conv_features, features_emb], dim=-1)
        return result
