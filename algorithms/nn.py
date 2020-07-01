import torch
import torch.nn as nn


# orthogonal init from ikostrikov
def init(
        module,
        weight_init=nn.init.orthogonal_,
        bias_init=nn.init.constant_,
        gain=1.0
):
    weight_init(module.weight, gain=gain)
    bias_init(module.bias, 0)
    return module


class ActorCriticTwoMLP(nn.Module):
    # two separate MLP for actor and critic, both with Tanh activation
    def __init__(
            self,
            observation_size, action_size, hidden_size,
            distribution
    ):
        super().__init__()

        gain = nn.init.calculate_gain('tanh')
        gain_policy = 0.01

        self.critic = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, 1))
        )

        if distribution == 'Beta':
            action_size *= 2
        elif distribution in ['TanhNormal', 'Normal']:
            self.actor_log_std = nn.Parameter(torch.full(action_size, -1.34))

        self.policy = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, 2 * action_size), gain=gain_policy)
        )
        self.distribution = distribution

    def forward(self, observation):
        value = self.critic(observation).squeeze(-1)
        if self.distribution in ['TanhNormal', 'Normal']:
            mean = self.policy(observation)
            log_std = self.actor_log_std.expand_as(mean)
            policy = torch.cat((mean, log_std), -1)
        else:
            policy = self.policy(observation)

        return policy, value


class ActorCriticAtari(nn.Module):
    # 2-layer CNN from DQN paper
    def __init__(
            self,
            action_size, distribution
    ):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        gain_policy = 0.01
        # (4, 84, 84) -> (32, 9, 9)
        self.conv = nn.Sequential(
            init(nn.Conv2d(4, 16, kernel_size=8, stride=4), gain=gain),
            nn.ReLU(),
            init(nn.Conv2d(16, 32, kernel_size=4, stride=2), gain=gain),
            nn.ReLU()
        )
        self.fe = init(nn.Linear(32 * 9 * 9, 256))

        if distribution == 'Beta':
            action_size *= 2
        elif distribution in ['TanhNormal', 'Normal']:
            self.actor_log_std = nn.Parameter(torch.full(action_size, -1.34))

        self.policy = init(nn.Linear(256, action_size), gain=gain_policy)
        self.value = init(nn.Linear(256, 1))
        self.distribution = distribution

    def forward(self, observation):
        # observation is an image of size (T, B, C, H, W)
        obs_size = observation.size()
        (time, batch), chw = obs_size[:2], obs_size[2:]
        observation = observation.view(time * batch, *chw)  # (T, B, C, H, W) -> (T * B, C, H, W)
        conv = self.conv(observation)
        flatten = conv.view(time, batch, -1)  # (T*B, C', H', W') -> (T, B, C' * H' * W')
        f = self.fe(flatten)
        if self.distribution in ['TanhNormal', 'Normal']:
            mean = self.policy(f)
            log_std = self.actor_log_std.expand_as(mean)
            policy = torch.cat((mean, log_std), -1)
        else:
            policy = self.policy(f)
        value = self.value(f).squeeze(-1)
        return policy, value


class ModelMLP(nn.Module):
    # predicts p(s'|s, a)
    def __init__(
            self,
            observation_size, action_size, hidden_size,
            predict_reward,
            predict_log_sigma
    ):
        super().__init__()

        self.predict_log_sigma = predict_log_sigma
        self.fe = nn.Sequential(
            nn.Linear(observation_size + action_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        if predict_reward:
            observation_size += 1
        if predict_log_sigma:
            # predict parameters for N(mu, sigma)
            self.last_layer = nn.Linear(hidden_size, 2 * observation_size)
        else:
            # predict just next state
            self.last_layer = nn.Linear(hidden_size, observation_size)

    def forward(self, observation, action):
        x = torch.cat((observation, action), dim=-1)
        x = self.fe(x)
        return self.last_layer(x)
