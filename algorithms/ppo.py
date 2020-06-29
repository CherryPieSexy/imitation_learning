from collections import defaultdict

import torch

from algorithms.policy_gradient import PolicyGradient


class PPO(PolicyGradient):
    # Is it possible to write nice, clean and well-readable PPO? I doubt it
    # this class implement core PPO methods: several train steps + policy and (optional) value clipping
    def __init__(
            self,
            *args,
            ppo_epsilon,  # clipping parameter
            ppo_n_epoch,
            ppo_mini_batch,
            use_ppo_value_loss,
            recompute_advantage,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ppo_epsilon = ppo_epsilon
        self.ppo_n_epoch = ppo_n_epoch
        self.ppo_mini_batch = ppo_mini_batch
        self.use_ppo_value_loss = use_ppo_value_loss
        self.recompute_advantage = recompute_advantage

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

        policy_loss = policy_loss.mean()
        return policy_loss.mean()

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
            observations, actions, rewards, not_done,
            policy_old,
            returns, advantage, step
    ):
        # 1) call nn, recompute returns and advantage if needed
        if self.recompute_advantage and step != 0:
            policy, value, returns, advantage = self._compute_returns(observations, rewards, not_done)
        else:
            policy, value = self.nn(observations[:-1])

        # 2) sample subset of indices to optimize on
        time, batch = actions.size()[:2]
        flatten_indices = torch.randint(
            time * batch,
            size=(self.ppo_mini_batch,),
            dtype=torch.long, device=self.device
        )
        # noinspection PyUnresolvedReferences
        row = flatten_indices // batch
        col = flatten_indices - batch * row

        # 3) calculate losses and optimize
        value_loss, policy_loss, entropy, loss = self._calc_losses(
            policy_old[row, col],
            policy[row, col], value[row, col],
            actions[row, col], returns[row, col], advantage[row, col]
        )
        grad_norm = self._optimize_loss(loss)
        result = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item(),
            'grad_norm': grad_norm
        }
        return result

    def train_on_rollout(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
        :return: dict of loss values, gradient norm
        """
        observations, actions, rewards, not_done = self._rollout_to_tensors(rollout)

        result = defaultdict(float)
        with torch.no_grad():
            policy_old, _, returns, advantage = self._compute_returns(
                observations, rewards, not_done
            )

        n = self.ppo_n_epoch
        for ppo_epoch in range(n):
            step_result = self._ppo_train_step(
                observations, actions, rewards, not_done,
                policy_old,
                returns, advantage, ppo_epoch
            )
            for key, value in step_result.items():
                result[key] += value / n

        return result
