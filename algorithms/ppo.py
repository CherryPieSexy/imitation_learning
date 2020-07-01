from collections import defaultdict

import torch
import numpy as np

from algorithms.policy_gradient import PolicyGradient


class PPO(PolicyGradient):
    """Proximal Policy Optimization implementation

    This class contains core PPO methods:
        several training epochs on one rollout,
        splitting rollout into mini-batches,
        policy and (optional) value clipping.
    In addition this class may use rollback policy loss
    and can recompute advantage
    """
    def __init__(
            self,
            *args,
            ppo_epsilon,
            use_ppo_value_loss,
            rollback_alpha,
            recompute_advantage,
            ppo_n_epoch,
            ppo_n_mini_batches,
            **kwargs
    ):
        """
        PPO algorithm class

        :param args: PolicyGradient class args
        :param ppo_epsilon:         float, policy (and optionally value) clipping parameter
        :param use_ppo_value_loss:  bool, switches value loss function
                                    from PPO-like clipped (True) or simple MSE (False)
        :param rollback_alpha:      float, policy-rollback loss parameter.
                                    Rollback is turned on if rollback_alpha > 0
        :param recompute_advantage: bool, if True the returns and advantage
                                    will be recomputed after each nn update
        :param ppo_n_epoch:         int, number of training epoch on one rollout
        :param ppo_n_mini_batches:  int, number of mini-batches into which
                                    the training data is divided during one epoch
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.ppo_epsilon = ppo_epsilon
        self.use_ppo_value_loss = use_ppo_value_loss
        self.recompute_advantage = recompute_advantage
        self.rollback_alpha = rollback_alpha

        self.ppo_n_epoch = ppo_n_epoch
        self.ppo_n_mini_batches = ppo_n_mini_batches

    def _policy_loss(self, policy_old, policy, actions, advantage):
        log_pi_for_actions_old = self.distribution.log_prob(policy_old, actions)
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        log_prob_ratio = log_pi_for_actions - log_pi_for_actions_old

        prob_ratio = log_prob_ratio.exp()
        if self.rollback_alpha > 0:
            policy_loss = self._rollback_loss(prob_ratio, advantage)
        else:
            policy_loss = self._clipped_loss(prob_ratio, advantage)

        policy_loss = policy_loss.mean()
        return policy_loss.mean()

    def _clipped_loss(self, prob_ratio, advantage):
        prob_ratio_clamp = torch.clamp(
            prob_ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon
        )

        surrogate_1 = prob_ratio * advantage
        surrogate_2 = prob_ratio_clamp * advantage
        policy_loss = torch.min(surrogate_1, surrogate_2)
        return policy_loss

    def _rollback_loss(self, prob_ratio, advantage):
        eps = self.ppo_epsilon
        alpha = self.rollback_alpha

        pos_adv_rollback = torch.where(
            prob_ratio <= 1.0 + eps,
            prob_ratio,
            -alpha * prob_ratio + (1.0 + alpha) * (1.0 + eps)
        )
        neg_adv_rollback = torch.where(
            prob_ratio >= 1.0 - eps,
            prob_ratio,
            -alpha * prob_ratio + (1.0 + alpha) * (1.0 - eps)
        )
        policy_loss = advantage * torch.where(
            advantage >= 0,
            pos_adv_rollback,
            neg_adv_rollback
        )
        return policy_loss

    def _clipped_value_loss(self, values_old, values, returns):
        # clipped value loss, PPO-style
        clipped_value = values_old + torch.clamp(
            (values - values_old), -self.ppo_epsilon, self.ppo_epsilon
        )

        surrogate_1 = (values - returns) ** 2
        surrogate_2 = (clipped_value - returns) ** 2

        value_loss = 0.5 * torch.max(surrogate_1, surrogate_2)
        return value_loss.mean()

    @staticmethod
    def _mse_value_loss(values, returns):
        # simple MSE loss, works better than clipped PPO-style
        value_loss = 0.5 * ((values - returns) ** 2)
        return value_loss.mean()

    def _value_loss(self, values_old, values, returns):
        if self.use_ppo_value_loss:
            value_loss = self._clipped_value_loss(values_old, values, returns)
        else:
            value_loss = self._mse_value_loss(values, returns)
        return value_loss

    def _calc_losses(
            self,
            policy_old,
            policy, values, actions,
            returns, advantage
    ):
        # value_loss = self._value_loss(values_old, values, returns)
        value_loss = self._mse_value_loss(values, returns)
        policy_loss = self._policy_loss(policy_old, policy, actions, advantage)
        entropy = self.distribution.entropy(policy, actions).mean()
        loss = value_loss - policy_loss - self.entropy * entropy
        return value_loss, policy_loss, entropy, loss

    def _ppo_train_step(
            self,
            rollout_t, row, col,
            policy_old, returns, advantage
    ):
        observations, actions, rewards, not_done = rollout_t
        # 1) call nn, recompute returns and advantage if needed
        if self.recompute_advantage:  # one unnecessary call during the first train-op
            # to compute returns and advantage we have to call nn.forward(...) on full data
            policy, value, returns, advantage = self._compute_returns(observations, rewards, not_done)
            policy, value = policy[row, col], value[row, col]
        else:
            # here we can call nn.forward(...) only on interesting data
            policy, value = self.nn(observations[row, col])

        # 2) calculate losses and optimize
        value_loss, policy_loss, entropy, loss = self._calc_losses(
            policy_old[row, col],
            policy, value,
            actions[row, col], returns[row, col], advantage[row, col]
        )
        grad_norm = self._optimize_loss(loss)

        # 3) store training results in dict and return
        result = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item(),
            'grad_norm': grad_norm
        }
        return result

    def _ppo_epoch(
            self,
            rollout_t, time, batch,
            policy_old, returns, advantage
    ):
        # goes once trough rollout
        epoch_result = defaultdict(float)

        # 1) select indices to train on during epoch
        n_transitions = time * batch
        flatten_indices = np.arange(n_transitions)
        np.random.shuffle(flatten_indices)
        n_batch_train = n_transitions // self.ppo_n_mini_batches

        for start_id in range(0, n_transitions, n_batch_train):
            indices_to_train_on = flatten_indices[start_id:start_id + n_batch_train]
            row = indices_to_train_on // batch
            col = indices_to_train_on - batch * row

            step_result = self._ppo_train_step(
                rollout_t, row, col,
                policy_old, returns, advantage
            )

            for key, value in step_result.items():
                epoch_result[key] += value / n_batch_train

        return epoch_result

    def train_on_rollout(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
        :return: dict of loss values, gradient norm
        """
        observations, actions, rewards, not_done = self._rollout_to_tensors(rollout)
        time, batch = actions.size()[:2]
        rollout_t = (observations, actions, rewards, not_done)

        result = defaultdict(float)
        with torch.no_grad():
            policy_old, _, returns, advantage = self._compute_returns(
                observations, rewards, not_done
            )

        n = self.ppo_n_epoch
        for ppo_epoch in range(n):
            step_result = self._ppo_epoch(rollout_t, time, batch, policy_old, returns, advantage)
            for key, value in step_result.items():
                result[key] += value / n

        return result
