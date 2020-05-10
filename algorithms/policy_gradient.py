# base class for policy gradient algorithms (A2C and PPO)

import torch
from torch.nn.utils import clip_grad_norm_

from algorithms.policies import distributions_dict
from algorithms.nn import ActorCriticMLP, ActorCriticAtari


class PolicyGradientMinimal:
    # minimal version of the class below. Can only save/load nn and act
    def __init__(
            self,
            image_env,
            observation_size, action_size, hidden_size, device,
            distribution
    ):
        self.device = device
        self.distribution = distributions_dict[distribution]()
        if distribution in ['Beta', 'TanhNormal']:
            action_size *= 2
        categorical = distribution in ['Categorical', 'GumbelSoftmax']

        if image_env:
            self.nn = ActorCriticAtari(action_size, categorical)
        else:
            self.nn = ActorCriticMLP(
                observation_size, action_size, hidden_size,
                categorical
            )

        self.nn.to(device)

    def save(self, filename):
        state_dict = {
            'nn': self.nn.state_dict()
        }
        torch.save(state_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.nn.load_state_dict(checkpoint['nn'])

    def act(self, observations, deterministic=False):
        with torch.no_grad():
            policy, _ = self.nn(
                torch.tensor(observations, dtype=torch.float32, device=self.device)
            )
            action = self.distribution.sample(policy, deterministic)
        action = action.cpu().numpy()
        return action


class PolicyGradient:
    def __init__(
            self,
            image_env,
            observation_size, action_size, hidden_size, device,
            distribution, normalize_adv, returns_estimator,
            lr, gamma, entropy, clip_grad,
            gae_lambda=0.95
    ):
        """
        :param image_env: True or False
        :param observation_size, action_size, hidden_size, device: just nn parameters
        :param distribution: 'Categorical' or 'NormalFixed' or 'Beta'
        :param normalize_adv: True or False
        :param returns_estimator: '1-step', 'n-step', 'gae'
        :param lr, gamma, entropy, clip_grad: learning hyper-parameters
        :param gae_lambda: gae lambda, optional
        """
        self.device = device

        self.distribution = distributions_dict[distribution]()
        if distribution in ['Beta', 'TanhNormal']:
            action_size *= 2

        # policy, value = nn(obs)
        # policy.size() == (T, B, dim(A))
        # value.size() == (T, B)  - there is no '1' at the last dimension!
        categorical = distribution in ['Categorical', 'GumbelSoftmax']

        if image_env:
            self.nn = ActorCriticAtari(action_size, categorical)
        else:
            self.nn = ActorCriticMLP(
                observation_size, action_size, hidden_size,
                categorical=categorical
            )

        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)

        self.normalize_adv = normalize_adv
        self.returns_estimator = returns_estimator
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy = entropy
        self.clip_grad = clip_grad

    def save_policy(self, filename):
        """
        Saves only nn weights (aka policy) into a file
        :param filename: str
        """
        state_dict = {
            'nn': self.nn.state_dict()
        }
        torch.save(state_dict, filename)

    def save(self, filename):
        """
        Saves nn and optimizer into a file
        :param filename: str
        """
        state_dict = {
            'nn': self.nn.state_dict(),
            'opt': self.opt.state_dict()
        }
        torch.save(state_dict, filename)

    def load_policy(self):
        pass

    def load(self):
        pass

    def _get_state_dict(self):
        state_dict = {
            'nn': self.nn.state_dict(),
            'opt': self.opt.state_dict()
        }
        return state_dict

    def act(self, observations, deterministic=False):
        """
        :param observations: np.array of observation, shape = [T, B, dim(obs)]
        :param deterministic: True or False
        :return: action, np.array of shape = [T, B, dim(action)]
        """
        policy, _ = self.nn(
            torch.tensor(observations, dtype=torch.float32, device=self.device)
        )
        action = self.distribution.sample(policy, deterministic)
        # we can not use 'action.cpu().numpy()' here, because we want action to have gradient
        return action

    def _one_step_returns(self, next_values, rewards, not_done):
        returns = rewards + self.gamma * not_done * next_values
        return returns

    def _n_step_returns(self, next_values, rewards, not_done):
        rollout_len = rewards.size(0)
        last_value = next_values[-1]
        returns = []
        for t in reversed(range(rollout_len)):
            last_value = rewards[t] + self.gamma * not_done[t] * last_value
            returns.append(last_value)
        returns = torch.stack(returns[::-1])
        return returns

    def _gae(self, values, next_values, rewards, not_done):
        rollout_len = rewards.size(0)
        values = torch.cat([values, next_values[-1:]], dim=0)
        gae = 0
        returns = []
        for t in reversed(range(rollout_len)):
            delta = rewards[t] + self.gamma * not_done[t] * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
            returns.append(gae + values[t])
        returns = torch.stack(returns[::-1])
        return returns

    def _estimate_returns(self, values, next_values, rewards, not_done):
        with torch.no_grad():  # returns should not have gradients in any case!
            if self.returns_estimator == '1-step':
                returns = self._one_step_returns(next_values, rewards, not_done)
            elif self.returns_estimator == 'n-step':
                returns = self._n_step_returns(next_values, rewards, not_done)
            elif self.returns_estimator == 'gae':
                returns = self._gae(values, next_values, rewards, not_done)
            else:
                raise ValueError('unknown returns estimator')
        return returns.detach()

    @staticmethod
    def _normalize_advantages(advantages):
        # advantages normalized across batch dimension, I think this is correct
        mean = advantages.mean(dim=1, keepdim=True)
        std = advantages.std(dim=1, keepdim=True)
        advantages = (advantages - mean) / (std + 1e-8)
        return advantages

    def _optimize_loss(self, loss):
        self.opt.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(
            self.nn.parameters(), self.clip_grad
        )
        self.opt.step()
        return gradient_norm

    def _preprocess_rollout(self, rollout):
        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        observations, actions, rewards, is_done = list(map(to_tensor, rollout))
        policy, values = self.nn(observations)
        policy = policy[:-1]
        values, next_values = values[:-1], values[1:].detach()
        not_done = 1.0 - is_done

        returns = self._estimate_returns(values, next_values, rewards, not_done)
        advantages = (returns - values).detach()
        # for PPO advantage must be normalized here,
        # so we must feed values and returns separately to compute value loss

        if self.normalize_adv:
            advantages = self._normalize_advantages(advantages)

        rollout_t = (observations, actions, rewards, not_done)

        return rollout_t, policy, values, returns, advantages

    def train_on_rollout(self, *args, **kwargs):
        raise NotImplementedError
