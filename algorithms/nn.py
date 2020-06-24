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


class ActorCriticMLP(nn.Module):
    # just 3-layer MLP with relu activation and policy & value heads
    def __init__(
            self,
            observation_size, action_size, hidden_size
    ):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        gain_policy = 0.01

        self.fe = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.ReLU(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.ReLU()
        )

        self.policy = init(nn.Linear(hidden_size, action_size), gain=gain_policy)
        self.value = init(nn.Linear(hidden_size, 1))

    def forward(self, observation):
        f = self.fe(observation)
        return self.policy(f), self.value(f).squeeze(-1)


class ActorCriticTwoMLP(nn.Module):
    # two separate MLP for actor and critic, both with Tanh activation
    def __init__(
            self,
            observation_size, action_size, hidden_size
    ):
        super().__init__()

        gain = nn.init.calculate_gain('tanh')
        gain_policy = 0.01

        self.critic = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, 1))
        )

        self.actor = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, 2 * action_size), gain=gain_policy)
        )

    def forward(self, observation):
        policy = self.actor(observation)
        value = self.critic(observation).squeeze(-1)
        return policy, value


class ActorCriticAtari(nn.Module):
    # 2-layer CNN from DQN paper
    def __init__(
            self,
            action_size
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

        self.policy = init(nn.Linear(256, action_size), gain=gain_policy)
        self.value = init(nn.Linear(256, 1))

    def forward(self, observation):
        # observation is an image of size (T, B, C, H, W)
        obs_size = observation.size()
        (time, batch), chw = obs_size[:2], obs_size[2:]
        observation = observation.view(time * batch, *chw)  # (T, B, C, H, W) -> (T * B, C, H, W)
        conv = self.conv(observation)
        flatten = conv.view(time, batch, -1)  # (T*B, C', H', W') -> (T, B, C' * H' * W')
        f = self.fe(flatten)
        return self.policy(f), self.value(f).squeeze(-1)


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
