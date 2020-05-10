import torch.nn as nn


# orthogonal init from ikostrikov
def init(
        module,
        weight_init=nn.init.orthogonal_,
        bias_init=nn.init.constant_,
        gain=1.0
):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data, 0)
    return module


# In future we will use CNN and RNN, this should be indicated by class name
class ActorCriticMLP(nn.Module):
    # just 3-layer MLP with relu activation and policy & value heads
    def __init__(
            self,
            observation_size, action_size, hidden_size,
            categorical=False
    ):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        self.fe = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.ReLU(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.ReLU()
        )

        if categorical:
            gain = 0.01
        self.policy = init(nn.Linear(hidden_size, action_size), gain=gain)
        self.value = init(nn.Linear(hidden_size, 1))

    def forward(self, observation):
        f = self.fe(observation)
        return self.policy(f), self.value(f).squeeze(-1)


class ActorCriticAtari(nn.Module):
    # 2-layer CNN from DQN paper
    def __init__(
            self,
            action_size,
            categorical=False
    ):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        # (4, 84, 84) -> (32, 9, 9)
        self.conv = nn.Sequential(
            init(nn.Conv2d(4, 16, kernel_size=8, stride=4), gain=gain),
            nn.ReLU(),
            init(nn.Conv2d(16, 32, kernel_size=4, stride=2), gain=gain),
            nn.ReLU()
        )
        self.fe = init(nn.Linear(32 * 9 * 9, 256))

        if categorical:
            gain = 0.01
        self.policy = init(nn.Linear(256, action_size), gain=gain)
        self.value = init(nn.Linear(256, 1))

    def forward(self, observation):
        # observation is an image of size (T, B, C, H, W)
        obs_size = observation.size()
        (time, batch), chw = obs_size[:2], obs_size[2:]
        observation = observation.view(time * batch, *chw)
        conv = self.conv(observation)
        flatten = conv.view(time, batch, -1)
        f = self.fe(flatten)
        return self.policy(f), self.value(f).squeeze(-1)


class QNetwork(nn.Module):
    # 3-layer MLP with dueling architecture
    def __init__(self, observation_size, action_size, hidden_size):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
        )
        self.value_layer = nn.Linear(hidden_size, 1)
        self.advantage_layer = nn.Linear(hidden_size, action_size)

    def forward(self, observation):
        f = self.fe(observation)
        v = self.value_layer(f)
        a = self.advantage_layer(f)
        q = v - (a - a.mean(-1, keepdim=True))
        return q
