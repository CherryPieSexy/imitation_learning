import pickle

import torch
from torch.utils.data import Dataset

from algorithms.agents.base_agent import AgentTrain
from algorithms.distributions import continuous_distributions


class BehaviorCloning(AgentTrain):
    def __init__(self, *args, loss_type='mse', **kwargs):
        assert loss_type in ['mse', 'likelihood']
        if loss_type == 'mse':
            # this loss supported only for continuous distributions
            assert self.policy_distribution_str in continuous_distributions, \
                'mse loss supported only for continuous distributions'
            self._loss_fn = self._mean_loss
        else:
            self._loss_fn = self._log_prob_loss
        super().__init__(*args, **kwargs)

    def to_tensor(self, x):
        return x.to(torch.float32).to(self.device)

    def _mean_loss(self, policy, actions):
        # I hope this have some meaning
        mean, _ = self.policy_distribution.mean(policy)
        loss = 0.5 * (mean - actions) ** 2
        return loss

    def _log_prob_loss(self, policy, actions):
        log_pi_for_actions = self.policy_distribution.log_prob(policy, actions)
        loss = -log_pi_for_actions
        return loss

    def _train_fn(self, rollout):
        # here rollout is just observations and actions
        observations, actions, _ = rollout
        observations = self.to_tensor(observations).unsqueeze(1)
        actions = self.to_tensor(actions).unsqueeze(1)

        nn_result = self.actor_critic_nn(observations)
        policy = nn_result['policy']
        loss = self._loss_fn(policy, actions).mean()

        grad_norm = self._optimize_loss(loss)
        result = {
            'loss': loss.item(),
            'grad_norm': grad_norm
        }

        return result


class BCDataSet(Dataset):
    def __init__(self, demo_file):
        with open(demo_file, 'rb') as f:
            data = pickle.load(f)

        self.observations = []
        self.actions = []
        self.rewards = []

        observations, actions, rewards = zip(*data)

        for episode_num in range(len(observations)):
            for transition_num in range(len(observations[episode_num]) - 1):
                self.observations.append(observations[episode_num][transition_num])
                self.actions.append(actions[episode_num][transition_num])
                self.rewards.append(rewards[episode_num][transition_num])

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # add time dimension
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx]
        )
