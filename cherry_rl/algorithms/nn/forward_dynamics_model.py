import torch
import torch.nn as nn

from cherry_rl.algorithms.nn.actor_critic import init, MLP
from cherry_rl.algorithms.distributions import distributions_dict


class ForwardDynamicsModel(nn.Module):
    """
    Predicts obs[t+1] ~ P(.|obs[t], a).
    Supports only 'deterministic' and 'RNormal' distributions.
    """
    def forward(
            self,
            observation: torch.Tensor,
            action: torch.Tensor
    ):
        raise NotImplementedError

    def sample(
            self,
            observation: torch.Tensor,
            action: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        model_output = self.forward(observation, action)
        if self.state_distribution_str == 'deterministic':
            return model_output
        else:
            sample, log_prob = self.state_distribution.sample(model_output, deterministic)
            return sample


class ForwardDynamicsContinuousActionsModel(ForwardDynamicsModel):
    def __init__(
            self,
            observation_size: int,
            hidden_size: int,
            action_size: int,
            state_distribution_str: str,
            n_layers: int = 3, activation_str: str = 'tanh', output_gain: float = 1.0
    ):
        assert \
            state_distribution_str in ['deterministic', 'RNormal'],\
            "Only 'deterministic' and 'Normal' distributions are supported for forward dynamics model"
        super().__init__()

        self.mlp = MLP(
            observation_size + action_size, hidden_size, observation_size,
            n_layers=n_layers, activation_str=activation_str, output_gain=output_gain
        )
        self.state_distribution_str = state_distribution_str
        if state_distribution_str != 'deterministic':
            self.state_distribution = distributions_dict[state_distribution_str]()

    def forward(
            self,
            observation: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        observation_action = torch.cat([observation, action], dim=-1)
        return self.mlp(observation_action)


class ForwardDynamicsDiscreteActionsModel(ForwardDynamicsModel):
    def __init__(
            self,
            observation_size: int,
            hidden_size: int,
            action_size: int,
            action_distribution_str: str,  # one from [Categorical, Gumbel, Bernoulli]
            state_distribution_str: str,
            n_layers: int = 3, activation_str: str = 'tanh', output_gain: float = 1.0
    ):
        assert \
            state_distribution_str in ['deterministic', 'RNormal'], \
            "Only 'deterministic' and 'Normal' distributions are supported for forward dynamics model"
        super().__init__()

        self.observation_embedding = nn.Linear(observation_size, hidden_size)
        self.action_distribution_str = action_distribution_str
        if action_distribution_str == 'Bernoulli':
            self.action_embedding = init(nn.Linear(action_size, hidden_size))
        else:
            self.action_embedding = init(nn.Embedding(action_size, hidden_size))
        self.mlp = MLP(
            2 * hidden_size, hidden_size, observation_size,
            n_layers=n_layers - 1, activation_str=activation_str, output_gain=output_gain
        )

        self.state_distribution_str = state_distribution_str
        if state_distribution_str != 'deterministic':
            self.state_distribution = distributions_dict[state_distribution_str]()

    def forward(
            self,
            observation: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        observation_embedding = self.observation_embedding(observation)
        if self.action_distribution_str == 'Categorical':
            action = action.to(torch.long)
        action_embedding = self.action_embedding(action)
        observation_action_embedding = torch.cat([observation_embedding, action_embedding], dim=-1)
        return self.mlp(observation_action_embedding)
