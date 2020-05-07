from algorithms.policy_gradient import PolicyGradient


class A2C(PolicyGradient):
    """
    can act (with grad, it may be useful) and calculate loss on rollout.
    This class defines A2C-specific parts, such as
    loss-function and how to call it
    """

    def _policy_loss(self, policy, actions, advantages):
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantages.detach()
        return policy_loss.mean()

    def _main(self, policy, values, actions, returns, advantages):
        # simple for A2C: just call _calc_losses once and optimize them
        # slightly more complex for PPO:
        # it optimizes losses for several steps, but it returns same values

        value_loss = 0.5 * ((values - returns) ** 2).mean()
        policy_loss = self._policy_loss(policy, actions, advantages)
        entropy = self.distribution.entropy(policy).mean()
        loss = value_loss - policy_loss - self.entropy * entropy

        grad_norm = self._optimize_loss(loss)

        result = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item(),
            'grad_norm': grad_norm,
        }
        return result

    def train_on_rollout(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
        :return: dict of loss values, gradient norm, mean reward on rollout
        """

        rollout_t, policy, values, returns, advantages = self._preprocess_rollout(rollout)
        observations, actions, rewards, not_done = rollout_t
        result = self._main(policy, values, actions, returns, advantages)

        result['reward'] = rewards.mean().item()
        return result
