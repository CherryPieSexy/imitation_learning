from algorithms.agents.base_agent import AgentTrain


class A2C(AgentTrain):
    """
    can act (with grad, it may be useful) and calculate loss on rollout.
    This class defines A2C-specific parts, such as
    loss-function and how to call it
    """

    def _policy_loss(self, policy, actions, advantage):
        if self.normalize_adv:
            advantage = self._normalize_advantage(advantage)
        log_pi_for_actions = self.pi_distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantage.detach()
        return policy_loss

    @staticmethod
    def _value_loss(value, returns):
        return 0.5 * ((value - returns) ** 2)

    # I want to be able to easily modify only loss calculating.
    def calculate_loss(self, rollout_t, policy, value, returns, advantage):
        observations = rollout_t['observations']
        actions = rollout_t['actions']

        policy_loss = self._policy_loss(policy, actions, advantage).mean()
        value_loss = self._value_loss(value, returns).mean()
        entropy = self.pi_distribution.entropy(policy).mean()

        aug_loss, aug_dict = self._image_augmentation_loss(observations, policy, value)

        loss = value_loss - policy_loss - self.entropy * entropy + aug_loss

        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item(),
        }
        loss_dict.update(aug_dict)
        return loss, loss_dict

    # A2C basically consist of calculation and optimizing loss
    def _main(self, rollout_t):
        policy, value = self._get_policy_value(rollout_t['observations'])
        returns, advantage = self._compute_returns_advantage(
            value,
            rollout_t['rewards'],
            rollout_t['is_done']
        )

        if self.value_normalizer is not None:
            self.value_normalizer.update(returns)

        policy, value = policy[:-1], value[:-1]
        loss, result = self.calculate_loss(rollout_t, policy, value, returns, advantage)

        optimization_result = self._optimize_loss(loss)
        result.update(optimization_result)

        return result

    # I want to be able to easily modify data in rollout
    def _train_fn(self, rollout):
        rollout_t = self._rollout_to_tensor(rollout)
        # A2C requires no modifications of rollout

        result = self._main(rollout_t)
        return result
