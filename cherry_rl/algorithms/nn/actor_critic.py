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
    if hasattr(module, 'bias'):
        bias_init(module.bias, 0)
    return module


activations_dict = {
    'tanh': nn.Tanh, 'relu': nn.ReLU, 'elu': nn.ELU
}


class MLP(nn.Module):
    """
    MLP with n-layers and no activation at the end.
    """
    def __init__(
            self,
            input_size, hidden_size, output_size,
            n_layers,
            activation_str,
            output_gain
    ):
        super().__init__()
        gain = nn.init.calculate_gain(activation_str)
        activation = activations_dict[activation_str]

        if n_layers == 1:
            self.mlp = nn.Sequential(init(nn.Linear(input_size, output_size)))

        else:
            hidden_layers = []
            for _ in range(n_layers - 2):
                hidden_layers.append(init(nn.Linear(hidden_size, hidden_size), gain=gain))
                hidden_layers.append(activation())

            self.mlp = nn.Sequential(
                init(nn.Linear(input_size, hidden_size), gain=gain), activation(),
                *hidden_layers,
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
            n_layers=3, activation_str='tanh', output_gain=0.01
        )

    def forward(self, observation):
        mean = self.mean(observation)
        log_std = self.log_std.expand_as(mean)
        policy = torch.cat((mean, log_std), -1)
        return policy


def actor_critic_forward(actor_critic, observation):
    if type(observation) is dict:
        observation = torch.cat([value for _, value in observation.items()], dim=-1)

    # detach prevents gradient flowing into encoder.
    obs_pi = observation
    obs_v = observation
    if actor_critic.detach_actor:
        obs_pi = obs_pi.detach()
    if actor_critic.detach_critic:
        obs_v = obs_v.detach()

    policy = actor_critic.actor(obs_pi)
    value = actor_critic.critic(obs_v)

    result = {
        'policy': policy,
        'value': value,
    }
    return result


class ActorCriticTwoMLP(nn.Module):
    """
    Two separate MLPs for actor and critic.
    Both actor and critic may be detached from pytorch computation graph
    to prevent gradients flowing into encoder.
    """
    def __init__(
            self,
            input_size, hidden_size, action_size,
            n_layers=3,
            activation_str='tanh',
            critic_size=1,
            detach_actor=False, detach_critic=False
    ):
        super().__init__()

        self.actor = MLP(
            input_size, hidden_size, action_size,
            n_layers, activation_str, 0.01
        )
        self.critic = MLP(
            input_size, hidden_size, critic_size,
            n_layers, activation_str, 1.0
        )

        self.detach_actor = detach_actor
        self.detach_critic = detach_critic

    def forward(self, observation):
        return actor_critic_forward(self, observation)


class ActorCriticInitialized(nn.Module):
    """
    Class for already initialized actor & critic networks.
    """
    def __init__(
            self,
            actor, critic,
            detach_actor=False, detach_critic=False
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic

        self.detach_actor = detach_actor
        self.detach_critic = detach_critic

    def forward(self, observation):
        return actor_critic_forward(self, observation)
