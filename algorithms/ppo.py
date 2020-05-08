import torch
import numpy as np

from algorithms.policy_gradient import PolicyGradient


class PPO(PolicyGradient):
    # Is it possible to write nice, clean and well-readable PPO? I doubt it
    # this class implement core PPO methods: several train steps + policy and value clipping
    def __init__(self, *args, ppo_epsilon, ppo_n_epochs, ppo_mini_batch):
        super().__init__(*args)
        self.ppo_epsilon = ppo_epsilon
        self.ppo_n_epochs = ppo_n_epochs
        self.ppo_mini_batch = ppo_mini_batch

    def _policy_loss(self, policy_old, policy, actions, advantage):
        # clipped policy objective
        log_pi_for_actions_old = self.distribution.log_prob(policy_old, actions)
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        log_prob_ratio = log_pi_for_actions - log_pi_for_actions_old

        prob_ratio = log_prob_ratio.exp()
        prob_ratio_clamp = torch.clamp(
            prob_ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon
        )

        surrogate_1 = prob_ratio * advantage
        surrogate_2 = prob_ratio_clamp * advantage
        policy_loss = torch.min(surrogate_1, surrogate_2)

        return policy_loss.mean()

    def _value_loss(self, values_old, values, returns):
        # clipped value objective
        clipped_value = values_old + torch.clamp(
            (values - values_old), -self.ppo_epsilon, self.ppo_epsilon
        )

        surrogate_1 = (values - returns) ** 2
        surrogate_2 = (clipped_value - returns) ** 2

        value_loss = 0.5 * torch.max(surrogate_1, surrogate_2)

        return value_loss.mean()

    def _calc_losses(
            self,
            policy_old, values_old,
            policy, values, actions,
            returns, advantage
    ):
        value_loss = self._value_loss(values_old, values, returns)
        policy_loss = self._policy_loss(policy_old, policy, actions, advantage)
        entropy = self.distribution.entropy(policy).mean()
        loss = value_loss - policy_loss - self.entropy * entropy
        return value_loss, policy_loss, entropy, loss

    def _ppo_train_step(
            self,
            observations, actions,
            policy_old, values_old,
            returns, advantage
    ):
        # 1) sample subset of indices to optimize on, select rollout[indices]
        time, batch = actions.size()[:2]
        flatten_indices = torch.randint(
            time * batch,
            size=(self.ppo_mini_batch,),
            dtype=torch.long, device=self.device
        )
        # noinspection PyUnresolvedReferences
        row = flatten_indices // batch
        col = flatten_indices - batch * row

        # 2) calculate losses and optimize
        policy, values = self.nn(observations[row, col])
        # here we already have precomputed returns, advantages
        # and select only observations to optimize on,
        # so we do not need to drop last policy step like in A2C

        value_loss, policy_loss, entropy, loss = self._calc_losses(
            policy_old[row, col], values_old[row, col],
            policy, values,  # computed values already have desired shapes
            actions[row, col], returns[row, col], advantage[row, col]
        )
        grad_norm = self._optimize_loss(loss)
        result = (
            value_loss.item(), policy_loss.item(),
            entropy.item(), loss.item(), grad_norm
        )
        return np.array(result)

    def _main(
            self,
            observations, actions,
            policy_old, values_old,
            returns, advantage
    ):
        # both returns and advantages estimated once
        # at the beginning of iteration, i.e. with \theta_old,
        # inside 'loss_on_rollout' method

        # this method pretty similar to A2C but still quite different

        policy_old, values_old = policy_old.detach(), values_old.detach()
        # it is not great to have magic numbers like '5'.
        # What if in future there will be more values to return?
        ppo_result = np.zeros(5, dtype=np.float32)

        n = self.ppo_n_epochs
        for ppo_epoch in range(n):
            ppo_step_result = self._ppo_train_step(
                observations, actions,
                policy_old, values_old,
                returns, advantage
            )

            ppo_result += ppo_step_result

        result = {
            'value_loss': ppo_result[0],
            'policy_loss': ppo_result[1],
            'entropy': ppo_result[2],
            'loss': ppo_result[3],
            'grad_norm': ppo_result[4],
        }
        return result

    def train_on_rollout(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
        :return: dict of loss values, gradient norm, mean reward on rollout
        """

        with torch.no_grad():
            rollout_t, policy, values, returns, advantages = self._preprocess_rollout(rollout)
        observations, actions, rewards, not_done = rollout_t

        result = self._main(
            observations, actions,
            policy, values,
            returns, advantages
        )

        result['reward'] = rewards.mean().item()
        return result
