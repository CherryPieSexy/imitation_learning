from collections import defaultdict

import torch
import numpy as np

from utils.utils import time_it
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
        log_pi_for_actions_old = policy_old
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        log_prob_ratio = log_pi_for_actions - log_pi_for_actions_old
        log_prob_ratio.clamp_max_(20)

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

    @time_it
    def _ppo_train_step(
            self,
            step,
            rollout_t, row, col,
            policy_old, returns, advantage
    ):
        observations, actions, rewards, not_done = rollout_t
        # 1) call nn, recompute returns and advantage if needed
        # advantage always computed by training net,
        # so it is unnecessary to recompute adv at the first train-op
        if self.recompute_advantage and step != 0:
            # to compute returns and advantage we _have_ to call nn.forward(...) on full data
            policy, value, returns, advantage = self._compute_returns(observations, rewards, not_done)
            policy, value = policy[row, col], value[row, col]
        else:
            # here we can call nn.forward(...) only on interesting data
            # observations[row, col] has only batch dimension =>
            # need to unsqueeze and squeeze back
            policy, value = self.nn(observations[row, col].unsqueeze(0))
            policy, value = policy.squeeze(0), value.squeeze(0)

        # 2) calculate losses
        value_loss, policy_loss, entropy, loss = self._calc_losses(
            policy_old[row, col],
            policy, value,
            actions[row, col], returns[row, col], advantage[row, col]
        )

        # 3) calculate image_aug loss if needed
        if self.image_augmentation_alpha > 0.0:
            (policy_div, value_div), img_aug_time = self._augmentation_loss(
                policy.detach().unsqueeze(0),
                value.detach().unsqueeze(0),
                observations[row, col].unsqueeze(0)
            )
            loss += self.image_augmentation_alpha * (policy_div + value_div)
            upd = {
                'policy_div': policy_div.item(),
                'value_div': value_div.item(),
                'img_aug_time': img_aug_time
            }

        # optimize
        grad_norm = self._optimize_loss(loss)

        # 4) store training results in dict and return
        result = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item(),
            'grad_norm': grad_norm
        }
        if self.image_augmentation_alpha > 0.0:
            # noinspection PyUnboundLocalVariable
            result.update(upd)
        return result

    @time_it
    def _ppo_epoch(
            self,
            rollout_t, time, batch,
            policy_old, returns, advantage
    ):
        # goes once trough rollout
        epoch_result = defaultdict(float)
        mean_train_op_time = 0

        # select indices to train on during epoch
        n_transitions = time * batch
        flatten_indices = np.arange(n_transitions)
        np.random.shuffle(flatten_indices)

        num_batches = self.ppo_n_mini_batches
        # n_batch_train = number of elements to train-on, i.e. batch-size
        n_batch_train = n_transitions // num_batches

        for step, start_id in enumerate(range(0, n_transitions, n_batch_train)):
            indices_to_train_on = flatten_indices[start_id:start_id + n_batch_train]
            row = indices_to_train_on // batch
            col = indices_to_train_on - batch * row

            train_op_result, train_op_time = self._ppo_train_step(
                step,
                rollout_t, row, col,
                policy_old, returns, advantage
            )

            for key, value in train_op_result.items():
                epoch_result[key] += value / num_batches
            mean_train_op_time += train_op_time

        return epoch_result, mean_train_op_time / num_batches

    def _train_fn(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done, log_probs),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
               I want to store 'log_probs' inside rollout
               because online policy (i.e. the policy gathered rollout)
               may not be the trained policy
        :return: (loss_dict, time_dict)
        """
        # 'done' converts into 'not_done' inside '_rollout_to_tensors' method
        observations, actions, rewards, not_done, policy_old = self._rollout_to_tensors(rollout)
        policy_old = policy_old.squeeze(1)
        time, batch = actions.size()[:2]
        rollout_t = (observations, actions, rewards, not_done)

        result_log = defaultdict(float)

        mean_epoch_time = 0
        mean_train_op_time = 0
        time_log = dict()

        with torch.no_grad():
            _, _, returns, advantage = self._compute_returns(
                observations, rewards, not_done
            )

        n = self.ppo_n_epoch
        for ppo_epoch in range(n):
            (epoch_result, mean_train_op_time), epoch_time = self._ppo_epoch(
                rollout_t, time, batch, policy_old, returns, advantage
            )
            for key, value in epoch_result.items():
                result_log[key] += value / n

            mean_epoch_time += epoch_time

        time_log['mean_ppo_epoch'] = mean_epoch_time / n
        time_log['mean_train_op'] = mean_train_op_time / n
        if self.image_augmentation_alpha > 0.0:
            time_log['img_aug'] = result_log.pop('img_aug_time')

        return result_log, time_log
