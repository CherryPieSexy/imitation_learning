# base class for policy gradient algorithms (A2C and PPO)
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from utils.utils import time_it
from utils.batch_crop import batch_crop
from algorithms.kl_divergence import kl_divergence
from algorithms.distributions import distributions_dict
from algorithms.nn import ActorCriticTwoMLP, ActorCriticCNN
from algorithms.normalization import RunningMeanStd


class AgentInference:
    def __init__(
            self,
            image_env,
            observation_size, action_size, hidden_size, device,
            distribution
    ):
        self.device = device
        self.distribution = distributions_dict[distribution]()

        if image_env:
            self.nn = ActorCriticCNN(action_size, distribution)
        else:
            self.nn = ActorCriticTwoMLP(
                observation_size, action_size, hidden_size, distribution
            )

        self.nn.to(device)
        self.obs_normalizer = None

    def load_state_dict(self, state):
        self.nn.load_state_dict(state['nn'])

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.nn.load_state_dict(checkpoint['agent']['nn'])
        if 'obs_normalizer' in checkpoint:
            self.obs_normalizer = RunningMeanStd()
            self.obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])

    def train(self):
        self.nn.train()

    def eval(self):
        self.nn.eval()

    def act(self, observations, return_pi=False, deterministic=False):
        """
        :param observations: np.array of observation, shape = [T, B, dim(obs)]
        :param return_pi: True or False. If True then method return full pi, not just log(pi(a))
        :param deterministic: True or False
        :return: action, np.array of shape = [T, B, dim(action)]
        """
        if self.obs_normalizer is not None:
            mean, var = self.obs_normalizer.mean, self.obs_normalizer.var
            observations = (observations - mean) / np.sqrt(var + 1e-8)
        with torch.no_grad():
            policy, _ = self.nn(torch.tensor(observations, dtype=torch.float32, device=self.device))
        action, log_prob = self.distribution.sample(policy, deterministic)
        if return_pi:
            return action, policy
        else:
            return action, log_prob


class PolicyGradient:
    def __init__(
            self,
            image_env,
            observation_size, action_size, hidden_size, device,
            distribution, normalize_adv, returns_estimator,
            lr, gamma, entropy, clip_grad,
            gae_lambda=0.95, image_augmentation_alpha=0.0
    ):
        """
        :param image_env:
        :param observation_size, action_size, hidden_size, device: just nn parameters
        :param distribution: any from 'distributions.py' file
        :param normalize_adv: True or False
        :param returns_estimator: '1-step', 'n-step', 'gae'
        :param lr, gamma, entropy, clip_grad: learning hyper-parameters
        :param gae_lambda: gae lambda, optional
        :param image_augmentation_alpha: if > 0 then additional
                                         alpha * D_KL(pi, pi_aug) loss term will be used
        """
        self.device = device

        self.distribution_str = distribution
        self.distribution = distributions_dict[distribution]()

        # policy, value = nn(obs)
        # policy.size() == (T, B, dim(A) or 2 * dim(A))
        # value.size() == (T, B)  - there is no '1' at the last dimension!

        if image_env:
            self.nn = ActorCriticCNN(action_size, distribution)
        else:
            self.nn = ActorCriticTwoMLP(
                observation_size, action_size, hidden_size, distribution
            )

        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)
        self.training = True

        self.normalize_adv = normalize_adv
        self.returns_estimator = returns_estimator
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy = entropy
        self.clip_grad = clip_grad
        self.image_augmentation_alpha = image_augmentation_alpha

    def state_dict(self):
        state_dict = {
            'nn': self.nn.state_dict(),
            'opt': self.opt.state_dict()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.nn.load_state_dict(state_dict['nn'])
        self.opt.load_state_dict(state_dict['opt'])

    def save(self, filename, obs_normalizer=None, reward_normalizer=None):
        """
        Saves nn and optimizer into a file
        :param filename: str
        :param obs_normalizer: RunningMeanStd object
        :param reward_normalizer: RunningMeanStd object
        """
        state_dict = {
            'nn': self.nn.state_dict(),
            'opt': self.opt.state_dict(),
        }
        if obs_normalizer is not None:
            state_dict['obs_normalizer'] = obs_normalizer.state_dict()
        if reward_normalizer is not None:
            state_dict['reward_normalizer'] = reward_normalizer.state_dict()
        torch.save(state_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.nn.load_state_dict(checkpoint['agent']['nn'])
        self.opt.load_state_dict(checkpoint['agent']['opt'])

    def train(self):
        self.training = True
        self.nn.train()

    def eval(self):
        self.training = False
        self.nn.eval()

    def act(self, observations, return_pi=False, deterministic=False):
        """
        :param observations: np.array of observation, shape = [T, B, dim(obs)]
        :param return_pi: True or False. If True then method return full pi, not just log(pi(a))
        :param deterministic: True or False
        :return: action, torch.Tensor of shape = [T, B, dim(action)], with gradients
        """
        policy, _ = self.nn(
            torch.tensor(observations, dtype=torch.float32, device=self.device)
        )
        action, log_prob = self.distribution.sample(policy, deterministic)
        # we can not use 'action.cpu().numpy()' here, because we want action to have gradient
        if return_pi:
            return action, policy
        else:
            return action, log_prob

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

    def _compute_returns(self, observations, rewards, not_done):
        policy, value = self.nn(observations)
        policy = policy[:-1]
        value, next_value = value[:-1], value[1:].detach()

        returns = self._estimate_returns(value, next_value, rewards, not_done)
        advantage = (returns - value).detach()
        if self.normalize_adv:
            advantage = self._normalize_advantages(advantage)
        return policy, value, returns, advantage

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _rollout_to_tensors(self, rollout):
        observations, actions, rewards, is_done, policy_old = list(map(self.to_tensor, rollout))
        not_done = 1.0 - is_done
        return observations, actions, rewards, not_done, policy_old

    def _train_fn(self, rollout):
        raise NotImplementedError

    @time_it
    def _augmentation_loss(self, policy, value, observations):
        observations_aug = batch_crop(observations)
        policy_aug, value_aug = self.nn(observations_aug)
        policy_div = kl_divergence(self.distribution_str, policy, policy_aug).mean()
        value_div = 0.5 * ((value - value_aug) ** 2).mean()
        return policy_div, value_div

    def train_on_rollout(self, rollout):
        train_fn_result, train_fn_time = time_it(self._train_fn)(rollout)
        if isinstance(train_fn_result, tuple):
            result_log, time_log = train_fn_result
        else:  # i.e. result is one dict
            result_log = train_fn_result
            time_log = dict()
        time_log['train_on_rollout'] = train_fn_time
        return result_log, time_log
