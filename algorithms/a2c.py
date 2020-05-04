import torch
import torch.nn as nn

from .policies import Categorical, Beta, NormalFixedSigma


class NN(nn.Module):
    # just 3-layer MLP with relu
    def __init__(self, observation_size, action_size, hidden_size):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True)
        )
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observation):
        f = self.fe(observation)
        return self.policy(f), self.value(f).squeeze(-1)


class A2C:
    # can act (with no grad) and calculate loss on rollout
    def __init__(self,
                 observation_size, action_size, hidden_size, device,
                 distribution,
                 lr, gamma, entropy):
        self.device = device

        if distribution == 'Categorical':
            self.distribution = Categorical()
        elif distribution == 'NormalFixed':
            self.distribution = NormalFixedSigma()
        elif distribution == 'Beta':
            self.distribution = Beta()
            action_size *= 2
        else:
            raise ValueError(f'wrong distribution: {distribution}')

        self.nn = NN(observation_size, action_size, hidden_size)
        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)

        self.gamma = gamma
        self.entropy = entropy

    def act(self, observations):
        with torch.no_grad():
            logits, _ = self.nn(
                torch.tensor(observations, dtype=torch.float32)
            )
        action = self.distribution.sample(logits, False)
        return action.cpu().numpy()

    def _one_step_returns(self, next_values, rewards, not_done):
        returns = rewards + self.gamma * not_done * next_values
        return returns

    def _policy_loss(self, policy, actions, advantage):
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantage.detach()
        return policy_loss.mean()

    def loss_on_rollout(self, rollout):
        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        observations, actions, rewards, is_done = list(map(to_tensor, rollout))
        rewards *= 0.1
        policy, values = self.nn(observations)
        policy = policy[:-1]
        # actions = actions.to(torch.long)
        values, next_values = values[:-1], values[1:].detach()
        not_done = 1.0 - is_done

        returns = self._one_step_returns(next_values, rewards, not_done)
        advantage = returns - values

        value_loss = (advantage ** 2).mean()
        policy_loss = self._policy_loss(policy, actions, advantage)
        entropy = self.distribution.entropy(policy).mean()

        self.opt.zero_grad()
        loss = value_loss - policy_loss - self.entropy * entropy
        loss.backward()
        self.opt.step()
