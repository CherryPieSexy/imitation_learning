import torch
import torch.nn as nn

from cherry_rl.algorithms.nn.actor_critic import MLP
from cherry_rl.algorithms.distributions import distributions_dict


class InverseDynamicsModel(nn.Module):
    def __init__(
            self,
            observation_size, hidden_size, action_size,
            distribution_str,
            n_layers=3, activation_str='tanh', output_gain=1.0
    ):
        super().__init__()
        self.mlp = MLP(
            observation_size * 2, hidden_size, action_size,
            n_layers=n_layers, activation_str=activation_str, output_gain=output_gain
        )
        self.action_distribution_str = distribution_str
        if distribution_str != 'deterministic':
            self.action_distribution = distributions_dict[distribution_str]()

    def forward(self, current_observation, next_observation):
        cat_observation = torch.cat([current_observation, next_observation], dim=-1)
        return self.mlp(cat_observation)
