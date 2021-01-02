import torch
import torch.nn as nn

from algorithms.nn.actor_critic import init


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
    def __init__(self, input_channels=4):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        # (input_channels, 42, 42) -> (32, 4, 4)
        self.conv = nn.Sequential(
            init(nn.Conv2d(input_channels, 32, kernel_size=3, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2), gain=gain), nn.ReLU()
        )

    def forward(self, observation):
        return cnn_forward(self.conv, observation)


class DeepConvEncoder(nn.Module):
    """
    "Deep" 6-layer CNN with ReLU activation for images of size (input_channels, 96, 96).
    """
    def __init__(self, input_channels=4):
        super().__init__()
        gain = nn.init.calculate_gain('relu')
        # (input_channels, 96, 96) -> (256, 1, 1)
        self.conv = nn.Sequential(
            init(nn.Conv2d(input_channels, 8, kernel_size=4, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(8, 16, kernel_size=3, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(16, 32, kernel_size=3, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(32, 64, kernel_size=3, stride=2), gain=gain), nn.ReLU(),
            init(nn.Conv2d(64, 128, kernel_size=3, stride=1), gain=gain), nn.ReLU(),
            init(nn.Conv2d(128, 256, kernel_size=3, stride=1), gain=gain), nn.ReLU()
        )

    def forward(self, observation):
        return cnn_forward(self.conv, observation)


class TwoLayerActorCritic(nn.Module):
    """
    Two 2-layer MLPs (actor and critic) with ELU activation attached to conv encoder.
    Both actor and critic may be detached from pytorch computation graph
    to prevent gradients flowing into encoder.
    """
    def __init__(
            self,
            input_size, hidden_size, action_size, distribution,
            detach_actor=False, detach_critic=False
    ):
        super().__init__()
        gain = nn.init.calculate_gain('relu')
        gain_policy = 0.01
        self.detach_policy = detach_actor
        self.detach_critic = detach_critic
        if distribution in ['Beta', 'TanhNormal', 'Normal']:
            action_size *= 2

        self.critic = nn.Sequential(
            init(nn.Linear(input_size, hidden_size), gain=gain), nn.ELU(),
            init(nn.Linear(hidden_size, 1))
        )
        self.actor = nn.Sequential(
            init(nn.Linear(input_size, hidden_size), gain=gain), nn.ELU(),
            init(nn.Linear(hidden_size, action_size), gain=gain_policy)
        )

    def forward(self, observation):
        if type(observation) is dict:
            observation = torch.cat([value for _, value in observation.items()], dim=-1)

        # detach prevents gradient flowing into encoder.
        obs_v = obs_pi = observation
        if self.detach_critic:
            obs_v = obs_v.detach()
        if self.detach_policy:
            obs_pi = obs_pi.detach()

        value = self.critic(obs_v)
        policy = self.actor(obs_pi)
        return {'policy': policy, 'value': value}
