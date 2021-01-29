from algorithms.agents.agent_train import AgentTrain


class A2C(AgentTrain):
    """
    can act (with grad, it may be useful) and calculate loss on rollout.
    This class defines A2C-specific parts, such as
    loss-function and how to call it
    """
    def _policy_loss(self, policy, actions, advantage):
        if self.normalize_adv:
            advantage = self._normalize_advantage(advantage)
        log_pi_for_actions = self.model.pi_distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantage.detach()
        return policy_loss

    @staticmethod
    def _value_loss(value, returns):
        difference_2 = (value - returns) ** 2
        value_loss = 0.5 * (difference_2.mean(-1))
        return value_loss

    def calculate_loss(self, rollout_t, policy, value, mask):
        actions = rollout_t['actions']
        returns = rollout_t['returns']
        advantage = rollout_t['advantage']

        policy_loss = self._average_loss(self._policy_loss(policy, actions, advantage), mask)
        value_loss = self._average_loss(self._value_loss(value, returns), mask)
        entropy = self._average_loss(self.model.pi_distribution.entropy(policy), mask)

        # aug_loss, aug_dict = self._image_augmentation_loss(observations, policy, value)
        aug_loss, aug_dict = 0.0, dict()

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
    def _main(self, rollout):
        rollout_t, memory, mask = rollout.as_dict, rollout.memory, rollout.mask

        policy, value, _ = self.model(rollout_t['observations'], memory)
        returns, advantage = self._compute_returns_advantage(
            value,
            rollout_t['rewards'],
            rollout_t['is_done']
        )
        rollout.set('returns', returns)
        rollout.set('advantage', advantage)

        if self.model.value_normalizer is not None:
            self.model.value_normalizer.update(returns, mask)

        policy, value = policy[:-1], value[:-1]
        loss, result = self.calculate_loss(
            rollout.as_dict, policy, value, mask
        )

        optimization_result = self._optimize_loss(loss)
        result.update(optimization_result)

        return result

    # I want to be able to easily modify data in rollout
    def _train_fn(self, rollout):
        # A2C requires no modifications of rollout
        result = self._main(rollout)
        return result
