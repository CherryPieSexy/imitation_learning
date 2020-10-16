from collections import defaultdict

import torch
import numpy as np

from utils.utils import time_it
from algorithms.agents.base_agent import AgentTrain


class PPO(AgentTrain):
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
            ppo_n_epoch,
            ppo_n_mini_batches,
            ppo_epsilon=0.2,
            use_ppo_value_loss=False,
            rollback_alpha=0.05,
            recompute_advantage=False,
            **kwargs
    ):
        """
        PPO algorithm class

        :param args: PolicyGradient class args
        :param ppo_n_epoch:         int, number of training epoch on one rollout
        :param ppo_n_mini_batches:  int, number of mini-batches into which
                                    the training data is divided during one epoch
        :param ppo_epsilon:         float, policy (and optionally value) clipping parameter
        :param use_ppo_value_loss:  bool, switches value loss function
                                    from PPO-like clipped (True) or simple MSE (False).
                                    Currently only MSE loss is supported.
        :param rollback_alpha:      float, policy-rollback loss parameter.
                                    Rollback is turned on if rollback_alpha > 0
        :param recompute_advantage: bool, if True the returns and advantage
                                    will be recomputed after each nn update
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
        log_pi_for_actions = self.policy_distribution.log_prob(policy, actions)
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

    def _clipped_value_loss(self, value_old, value, returns):
        # clipped value loss, PPO-style
        clipped_value = value_old + torch.clamp(
            (value - value_old), -self.ppo_epsilon, self.ppo_epsilon
        )

        surrogate_1 = (value - returns) ** 2
        surrogate_2 = (clipped_value - returns) ** 2

        return 0.5 * torch.max(surrogate_1, surrogate_2).mean()

    @staticmethod
    def _mse_value_loss(value, returns):
        return 0.5 * ((value - returns) ** 2).mean()

    def _value_loss(self, value_old, value, returns):
        if self.use_ppo_value_loss:
            value_loss = self._clipped_value_loss(value_old, value, returns)
        else:
            value_loss = self._mse_value_loss(value, returns)
        return value_loss

    def calculate_loss(self, rollout_t, policy, value, returns, advantage):
        observations = rollout_t['observations']
        actions = rollout_t['actions']
        policy_old = rollout_t['log_prob']
        value_old = rollout_t['value']

        policy_loss = self._policy_loss(policy_old, policy, actions, advantage)
        value_loss = self._value_loss(value_old, value, returns)
        # value_loss = self._mse_value_loss(value, returns)
        entropy = self.policy_distribution.entropy(policy, actions).mean()

        aug_loss, aug_dict = self._image_augmentation_loss(policy, value, observations)

        loss = value_loss - policy_loss - self.entropy * entropy + aug_loss
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item()
        }
        loss_dict.update(aug_dict)
        return loss, loss_dict

    @time_it
    def _ppo_train_step(
            self,
            step,
            rollout_t, row, col,
            returns, advantage
    ):
        observations = rollout_t['observations']

        # 1) call nn, recompute returns and advantage if needed
        # advantage always computed by training net,
        # so it is unnecessary to recompute adv at the first train-op
        if self.recompute_advantage and step != 0:
            # to compute returns and advantage we _have_ to call nn.forward(...) on full data
            policy, value, returns, advantage = self._compute_returns(
                observations,
                rollout_t['rewards'],
                1.0 - rollout_t['is_done']
            )
            policy, value = policy[row, col], value[row, col]
        else:
            # here we can call nn.forward(...) only on interesting data
            # observations[row, col] has only batch dimension =>
            # need to unsqueeze and squeeze back
            nn_result = self.actor_critic_nn(observations[row, col].unsqueeze(0))
            policy, value = nn_result['policy'], nn_result['value']
            policy, value = policy.squeeze(0), value.squeeze(0)

        # 2) calculate losses
        loss, result = self.calculate_loss(
            self._select_by_row_col(rollout_t, row, col),
            policy, value,
            returns[row, col], advantage[row, col]
        )

        # 3) optimize
        optimization_result = self.optimize_loss(loss)
        result.update(optimization_result)

        return result

    @staticmethod
    def _select_by_row_col(tensor_dict, row, col):
        selected = {k: v[row, col] for k, v in tensor_dict.items()}
        return selected

    @time_it
    def _ppo_epoch(
            self,
            rollout_t, returns, advantage
    ):
        # goes once trough rollout
        epoch_result = defaultdict(float)
        mean_train_op_time = 0

        # select indices to train on during epoch
        time, batch = rollout_t['actions'].size()[:2]
        n_transitions = time * batch
        flatten_indices = np.arange(n_transitions)
        np.random.shuffle(flatten_indices)

        num_batches = self.ppo_n_mini_batches
        batch_size = n_transitions // num_batches

        for step, start_id in enumerate(range(0, n_transitions, batch_size)):
            indices_to_train_on = flatten_indices[start_id:start_id + batch_size]
            row = indices_to_train_on // batch
            col = indices_to_train_on - batch * row

            train_op_result, train_op_time = self._ppo_train_step(
                step,
                rollout_t, row, col,
                returns, advantage
            )

            for key, value in train_op_result.items():
                epoch_result[key] += value / num_batches
            mean_train_op_time += train_op_time

        return epoch_result, mean_train_op_time / num_batches

    def _main(self, rollout_t):
        """
        It may look a little bit complicated and unreasonable
        to make this method perform several training epoch
        instead of call 'agent._train_on_rollout(...)' outside several times,
        but with this approach I have to put rollout on device and
        compute returns and advantage only once (if option 'recompute_advantage' is disabled),
        which is slow with current GAE implementation.

        :param rollout_t:
        :return:
        """
        result_log = defaultdict(float)

        mean_epoch_time = 0
        mean_train_op_time = 0
        time_log = dict()

        with torch.no_grad():
            _, _, returns, advantage = self._compute_returns(
                rollout_t['observations'],
                rollout_t['rewards'],
                1.0 - rollout_t['is_done']
            )

        n = self.ppo_n_epoch
        for ppo_epoch in range(n):
            (epoch_result, mean_train_op_time), epoch_time = self._ppo_epoch(
                rollout_t, returns, advantage
            )
            for key, value in epoch_result.items():
                result_log[key] += value / n

            mean_epoch_time += epoch_time

        time_log['mean_ppo_epoch'] = mean_epoch_time / n
        time_log['mean_train_op'] = mean_train_op_time / n
        return result_log, time_log

    def _train_fn(self, rollout):
        """
        I moved all logic inside '_main(...)' method except for moving data on tensors
        and computing loss itself, which is done by 'calculate_loss(...)' method.
        I did this to make code more modular and in particular to have direct access
        to main algorithm function (i.e. loss computing and optimizing) for inheritance.
        It is useful if one want to add rollout pre-processing (for RND, for example)
        or modify only part of full loss

        :param rollout: tuple (observations, actions, rewards, is_done, log_probs),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
               I want to store 'log_probs' inside rollout
               because online policy (i.e. the policy gathered rollout)
               may differ from the trained policy
        :return: (loss_dict, time_dict)
        """
        rollout_t = self._rollout_to_tensor(rollout)
        # PPO requires no modifications on rollout
        result = self._main(rollout_t)

        # it is ok to return tuple of logs here, base class will handle this
        return result

    # TODO: recurrence:
    #  1) masking in _train_fn
    #  2) loss averaging, i.e. (mask * loss).sum() / mask.sum()
    #  3) index select fn, select indices randomly (like now) for feed-forward model,
    #  4) select only rows for recurrent model. Look how it is done in iKostrikov repo

    @staticmethod
    def _mask_after_done(done):
        pad = torch.zeros(done.size(1), dtype=torch.float32, device=done.device)
        done_sum = torch.cumsum(done, dim=0)[:-1]
        done_sum = torch.cat([pad, done_sum], dim=0)
        # noinspection PyTypeChecker
        mask = 1.0 - done_sum.clamp_max(1.0)
        return mask
