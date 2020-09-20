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
            # self.actor_log_std = nn.Parameter(torch.full((action_size,), -1.34))
            self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        elif distribution == 'RealNVP':
            gain_policy = gain

        self.policy = nn.Sequential(
            init(nn.Linear(observation_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.Tanh(),
            init(nn.Linear(hidden_size, action_size), gain=gain_policy)
        )
        self.distribution = distribution

    def forward(self, observation):
        if self.distribution in ['TanhNormal', 'Normal']:
            mean = self.policy(observation)
            log_std = self.actor_log_std.expand_as(mean)
            policy = torch.cat((mean, log_std), -1)
        else:
            policy = self.policy(observation)
        value = self.critic(observation).squeeze(-1)

        return policy, value


class ActorCriticCNN(nn.Module):
    # simple 3-layer CNN with ELU activation
    def __init__(
            self,
            action_size, distribution
    ):
        super().__init__()

        gain = nn.init.calculate_gain('relu')
        gain_policy = 0.01
        hidden_size = 128
        # (4, 42, 42) -> (32, 4, 4)
        self.conv = nn.Sequential(
            init(nn.Conv2d(4, 32, kernel_size=3, stride=2), gain=gain), nn.ELU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2), gain=gain), nn.ELU(),
            init(nn.Conv2d(32, 32, kernel_size=3, stride=2), gain=gain), nn.ELU()
        )
        self.fe = init(nn.Linear(512, hidden_size))

        if distribution in ['Beta', 'TanhNormal', 'Normal']:
            action_size *= 2

        self.policy = init(nn.Linear(hidden_size, action_size), gain=gain_policy)
        self.value = init(nn.Linear(hidden_size, 1))
        self.distribution = distribution

    def _cnn_forward(self, observation):
        # observation is an image of size (T, B, C, H, W)
        obs_size = observation.size()
        (time, batch), chw = obs_size[:2], obs_size[2:]
        observation = observation.view(time * batch, *chw)  # (T, B, C, H, W) -> (T * B, C, H, W)
        conv = self.conv(observation)
        flatten = conv.view(time, batch, -1)  # (T*B, C', H', W') -> (T, B, C' * H' * W')
        return flatten

    def _mlp_forward(self, flatten):
        if self.fe is not None:
            f = self.fe(flatten)
        else:
            f = flatten
        # if self.distribution in ['TanhNormal', 'Normal']:
        #     mean = self.policy(f)
        #     log_std = self.actor_log_std.expand_as(mean)
        #     policy = torch.cat((mean, log_std), -1)
        # else:
        policy = self.policy(f)
        value = self.value(f).squeeze(-1)
        return policy, value

    def forward(self, observation):
        flatten = self._cnn_forward(observation)
        policy, value = self._mlp_forward(flatten)
        return policy, value


class ActorCriticDeepCNN(ActorCriticCNN):
    def __init__(self, action_size, distribution, dropout=0.2):
        super().__init__(action_size, distribution)
        gain = nn.init.calculate_gain('relu')
        gain_policy = 0.01

        self.conv = nn.Sequential(  # input shape (4, 96, 96)
            init(nn.Conv2d(4, 8, kernel_size=4, stride=2), gain=gain), nn.ELU(),
            init(nn.Conv2d(8, 16, kernel_size=3, stride=2), gain=gain), nn.ELU(),
            init(nn.Conv2d(16, 32, kernel_size=3, stride=2), gain=gain), nn.ELU(),
            init(nn.Conv2d(32, 64, kernel_size=3, stride=2), gain=gain), nn.ELU(),
            init(nn.Conv2d(64, 128, kernel_size=3, stride=1), gain=gain), nn.ELU(),
            init(nn.Conv2d(128, 256, kernel_size=3, stride=1), gain=gain), nn.ELU()
        )  # output shape (256, 1, 1)
        self.fe = nn.Dropout(dropout)
        self.value = nn.Sequential(
            init(nn.Linear(256, 100), gain=gain), nn.ELU(),
            init(nn.Linear(100, 1))
        )
        if distribution in ['Beta', 'WideBeta', 'TanhNormal', 'Normal']:
            action_size *= 2
        self.policy = nn.Sequential(
            init(nn.Linear(256, 100), gain=gain), nn.ELU(),
            init(nn.Linear(100, action_size), gain=gain_policy)
        )


def init_actor_critic(nn_type, nn_args):
    neural_networks = {
        'MLP': ActorCriticTwoMLP,
        'CNN': ActorCriticCNN,
        'DeepCNN': ActorCriticDeepCNN,
    }
    return neural_networks[nn_type](**nn_args)
