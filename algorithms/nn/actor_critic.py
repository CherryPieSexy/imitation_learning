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


activation_dict = {
    'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU
}


class MLP(nn.Module):
    """
    3-layer MLP with no activation at the end.
    """
    def __init__(
            self,
            input_size, hidden_size, output_size,
            activation_str='tanh', output_gain=1.0
    ):
        super().__init__()
        gain = nn.init.calculate_gain(activation_str)
        activation = activation_dict[activation_str]

        self.mlp = nn.Sequential(
            init(nn.Linear(input_size, hidden_size), gain=gain), activation(),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), activation(),
            init(nn.Linear(hidden_size, output_size), gain=output_gain)
        )

    def forward(self, x):
        return self.mlp(x)


class ActorIndependentSigma(nn.Module):
    """
    Actor with std independent of observation.
    Not used now since dependent log-std works ok.
    """
    def __init__(self, observation_size, hidden_size, action_size):
        super().__init__()

        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.mean = MLP(
            observation_size, hidden_size, action_size,
            output_gain=0.01
        )

    def forward(self, observation):
        mean = self.mean(observation)
        log_std = self.log_std.expand_as(mean)
        policy = torch.cat((mean, log_std), -1)
        return policy


class ActorCriticTwoMLP(nn.Module):
    """
    Two separate MLP for actor and critic, both with Tanh activation.
    """
    def __init__(
            self,
            observation_size, hidden_size, action_size,
            distribution,
            critic_dim=1
    ):
        super().__init__()
        self.critic = MLP(observation_size, hidden_size, critic_dim)

        if distribution in ['Beta', 'TanhNormal', 'Normal']:
            action_size *= 2
        self.actor = MLP(observation_size, hidden_size, action_size, output_gain=0.01)

        self.distribution = distribution

    def forward(self, observation):
        policy = self.actor(observation)
        value = self.critic(observation)

        result = {
            'policy': policy,
            'value': value,
        }
        return result
