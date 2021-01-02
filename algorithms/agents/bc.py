import pickle

import torch
from torch.utils.data import Dataset

from algorithms.agents.base_agent import AgentTrain
from algorithms.distributions import continuous_distributions


class BehaviorCloningAgent(AgentTrain):
    def __init__(self, *args, loss_type='mse', **kwargs):
        super().__init__(*args, **kwargs)
        assert loss_type in ['mse', 'log_prob']
        self.loss_type = loss_type
        if loss_type == 'mse':
            # this loss supported only for continuous distributions
            assert self.pi_distribution_str in continuous_distributions, \
                'mse loss supported only for continuous distributions'
        super().__init__(*args, **kwargs)

    @staticmethod
    def _mean_loss(pi_action, demo_action):
        loss = 0.5 * (pi_action - demo_action) ** 2
        return loss

    def _log_prob_loss(self, policy, demo_action):
        log_pi_for_actions = self.pi_distribution.log_prob(policy, demo_action)
        loss = -log_pi_for_actions
        return loss

    def calculate_loss(self, policy, demo_action):
        if self.loss_type == 'mse':
            pi_action, _ = self.pi_distribution.sample(policy)
            bc_loss = self._mean_loss(pi_action, demo_action)
        else:
            bc_loss = self._log_prob_loss(policy, demo_action)

        bc_loss = bc_loss.mean()
        with torch.no_grad():
            entropy = self.pi_distribution.entropy(policy).mean()

        loss_dict = {
            'bc_loss': bc_loss.item(),
            'entropy': entropy.item()
        }
        return bc_loss, loss_dict

    def _main(self, rollout_t):
        observations, actions = rollout_t['observations'], rollout_t['actions']
        policy, _ = self._get_policy_value(observations)

        loss, result = self.calculate_loss(policy, actions)

        optimization_result = self._optimize_loss(loss)
        result.update(optimization_result)

        return result

    def _train_fn(self, rollout):
        # here 'rollout' have 3 keys: 'observations', 'actions', 'rewards'
        rollout_t = self._rollout_to_tensor(rollout)
        result = self._main(rollout_t)
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
        return {
            'observations': self.observations[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx]
        }
