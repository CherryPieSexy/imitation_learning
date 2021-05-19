from collections import defaultdict

import torch

from utils.utils import time_it
from algorithms.rollout_shuffler import get_data_generator
from algorithms.optimizers.actor_critic_optimizer import ActorCriticOptimizer


class PPO(ActorCriticOptimizer):
    """
    Proximal Policy Optimization.
    https://arxiv.org/abs/1707.06347

    This class contains core PPO methods:
        several training epochs on one rollout,
        splitting rollout into mini-batches,
        policy and (optional) value clipping.
    In addition this class may use rollback policy loss.
    """
    def __init__(
            self,
            *args,
            ppo_n_epoch,
            ppo_n_mini_batches,
            ppo_epsilon=0.2,
            use_ppo_value_loss=False,
            rollback_alpha=0.05,
            **kwargs
    ):
        """
        :param args: AgentTrain class args.
        :param ppo_n_epoch:         int, number of training epoch on one rollout.
        :param ppo_n_mini_batches:  int, number of mini-batches into which
                                    the training data is divided during one epoch.
        :param ppo_epsilon:         float, policy (and optionally value) clipping parameter.
        :param use_ppo_value_loss:  bool, switches value loss function.
                                    from PPO-like clipped (True) or simple MSE (False)
        :param rollback_alpha:      float, policy-rollback loss parameter.
                                    Rollback is turned on if rollback_alpha > 0.
        :param kwargs: AgentTrain class kwargs.
        """
        super().__init__(*args, **kwargs)
        self.ppo_epsilon = ppo_epsilon
        self.use_ppo_value_loss = use_ppo_value_loss
        self.rollback_alpha = rollback_alpha

        self.ppo_n_epoch = ppo_n_epoch
        self.ppo_n_mini_batches = ppo_n_mini_batches

    def _policy_loss(self, data_piece, policy, advantage):
        if self.normalize_adv:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        log_pi_for_actions_old = data_piece['log_prob']
        actions = data_piece['actions']

        log_pi_for_actions = self.model.pi_distribution.log_prob(policy, actions)
        log_prob_ratio = log_pi_for_actions - log_pi_for_actions_old
        log_prob_ratio.clamp_max_(20)

        prob_ratio = log_prob_ratio.exp()
        if self.rollback_alpha > 0:
            policy_loss = self._rollback_loss(prob_ratio, advantage)
        else:
            policy_loss = self._clipped_loss(prob_ratio, advantage)

        return policy_loss

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

    def _clipped_value_loss(self, value_old, value, value_target):
        # clipped value loss, PPO-style
        clipped_value = value_old + torch.clamp(
            (value - value_old), -self.ppo_epsilon, self.ppo_epsilon
        )

        surrogate_1 = (value - value_target) ** 2
        surrogate_2 = (clipped_value - value_target) ** 2
        clipped_mse = torch.max(surrogate_1, surrogate_2)

        return 0.5 * clipped_mse.mean(-1)

    @staticmethod
    def _mse_value_loss(value, value_target):
        difference_2 = (value - value_target) ** 2
        return 0.5 * (difference_2.mean(-1))

    def _value_loss(self, data_piece, value, value_target):
        if self.use_ppo_value_loss:
            value_old = data_piece['value']
            value_loss = self._clipped_value_loss(value_old, value, value_target)
        else:
            value_loss = self._mse_value_loss(value, value_target)
        return value_loss

    def calculate_loss(self, data_piece, policy, value):
        value_target = data_piece['value_target']
        advantage = data_piece['advantage']
        mask = data_piece['mask']

        policy_loss = self._average_loss(self._policy_loss(data_piece, policy, advantage), mask)
        value_loss = self._average_loss(self._value_loss(data_piece, value, value_target), mask)
        entropy = self._average_loss(self.model.pi_distribution.entropy(policy), mask)

        # aug_loss, aug_dict = self._image_augmentation_loss(rollout_t, policy, value)
        aug_loss, aug_dict = 0.0, dict()

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
    def _ppo_train_step(self, data_piece):
        # 1) call nn on accepted data:
        observations = data_piece['observations']
        memory = data_piece['memory']

        policy, value, _ = self.model(observations, memory)

        # 2) calculate losses:
        loss, result = self.calculate_loss(data_piece, policy, value)

        # 3) optimize:
        optimization_result = self.optimize_loss(loss)
        result.update(optimization_result)

        return result

    @time_it
    def _ppo_epoch(self, data_generator):
        epoch_result = defaultdict(float)
        mean_train_op_time = 0

        for data_piece in data_generator:
            train_op_result, train_op_time = self._ppo_train_step(data_piece)
            for key, value in train_op_result.items():
                epoch_result[key] += value / self.ppo_n_mini_batches
            mean_train_op_time += train_op_time

        return epoch_result, mean_train_op_time / self.ppo_n_mini_batches

    def _main(self, data_dict):
        data_dict = {
            k: (v.to(torch.float32) if type(v) is torch.Tensor else v)
            for k, v in data_dict.items()
        }
        recurrent = data_dict.pop('recurrent')

        result_log = defaultdict(float)
        mean_epoch_time = 0
        sum_mean_train_op_time = 0
        time_log = dict()

        with torch.no_grad():
            _, value, value_target, advantage = self.returns_estimator.policy_value_returns_adv(
                self.model, data_dict
            )
            data_dict['value_target'] = value_target
            data_dict['advantage'] = advantage

        if self.model.value_normalizer is not None:
            self.model.value_normalizer.update(value_target, data_dict['mask'])

        n = self.ppo_n_epoch
        for ppo_epoch in range(n):
            data_generator = get_data_generator(data_dict, recurrent, self.ppo_n_mini_batches)
            (epoch_result, mean_train_op_time), epoch_time = self._ppo_epoch(data_generator)
            for key, value in epoch_result.items():
                result_log[key] += value / n

            sum_mean_train_op_time += mean_train_op_time
            mean_epoch_time += epoch_time

        time_log['mean_ppo_epoch'] = mean_epoch_time / n
        time_log['mean_train_op'] = sum_mean_train_op_time / n
        return result_log, time_log

    def _train_fn(self, data_dict):
        # PPO requires no modifications on rollout
        result = self._main(data_dict)
        # it is ok to return tuple of logs here, base class will handle this
        return result
