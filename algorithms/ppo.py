import torch
import numpy as np

from algorithms.a2c import A2C


class PPO(A2C):
    # Is it possible to write nice, clean and well-readable PPO? I doubt it
    def __init__(self, *args, ppo_epsilon, ppo_n_epochs):
        super().__init__(*args)
        self.ppo_epsilon = ppo_epsilon
        self.ppo_n_epochs = ppo_n_epochs

    def _policy_loss_ppo(self, policy_old, policy, actions, advantage):
        # clipped policy objective
        log_pi_for_actions_old = self.distribution.log_prob(policy_old, actions)
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        log_prob_ratio = log_pi_for_actions - log_pi_for_actions_old

        prob_ratio = log_prob_ratio.exp()
        prob_ratio_clamp = torch.clamp(
            prob_ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon
        )

        if self.normalize_adv:
            advantage = self._normalize_advantage(advantage)

        surrogate_1 = prob_ratio * advantage
        surrogate_2 = prob_ratio_clamp * advantage
        policy_loss = torch.min(surrogate_1, surrogate_2)

        return policy_loss.mean()

    def _value_loss_ppo(self, values_old, values, returns):
        # clipped value objective
        clipped_value = values_old + torch.clamp(
            (values - values_old)-self.ppo_epsilon, self.ppo_epsilon
        )

        surrogate_1 = (values - returns) ** 2
        surrogate_2 = (clipped_value - returns) ** 2

        value_loss = 0.5 * torch.max(surrogate_1, surrogate_2)

        return value_loss.mean()

    def _calc_losses_ppo(
            self,
            policy_old, values_old,
            policy, values, actions,
            returns, advantage
    ):
        value_loss = self._value_loss_ppo(values_old, values, returns)
        policy_loss = self._policy_loss_ppo(policy_old, policy, actions, advantage)
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
        indices = np  # TODO: sample indices here!
        # TODO: how to choose mini-batch size here?
        #  in paper: T * T >= num_ppo_epoch
        #  in ikosrikov: mini_batch = T * B / num_ppo_epoch
        #  may be I will do the same? Specify num_ppo_epoch outside
        #  and calculate mini_batch here? PPO should be PinA for Pain in the Ass
        # 2) calculate losses and optimize
        # TODO: select indices here!
        policy, values = self.nn(observations)
        policy = policy[:-1]
        value_loss, policy_loss, entropy, loss = self._calc_losses_ppo(
            policy_old, values_old,
            policy, values, actions,
            returns, advantage
        )
        grad_norm = self._optimize_loss(loss)
        result = (
            value_loss.item(), policy_loss.item(),
            entropy.item(), loss.item(), grad_norm
        )
        return np.array(result)

    def _main(
            self,
            time, batch,
            rollout, policy_old, values_old,
            returns, advantage
    ):
        # both returns and advantages estimated once
        # at the beginning of iteration, i.e. with \theta_old,
        # inside 'loss_on_rollout' method

        # this method pretty similar to A2C but still quite different

        observations, actions, rewards, not_done = rollout

        # it is not great to have magic numbers like '6'.
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

        return ppo_result
