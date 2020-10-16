from algorithms.agents.base_agent import AgentTrain


class A2C(AgentTrain):
    """
    can act (with grad, it may be useful) and calculate loss on rollout.
    This class defines A2C-specific parts, such as
    loss-function and how to call it
    """

    def _policy_loss(self, policy, actions, advantage):
        log_pi_for_actions = self.policy_distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantage.detach()
        return policy_loss.mean()

    @staticmethod
    def _value_loss(value, returns):
        return 0.5 * ((value - returns) ** 2).mean()

    # I want to be able to easily modify only loss calculating.
    def calculate_loss(self, observations, actions, policy, value, returns, advantage):
        policy_loss = self._policy_loss(policy, actions, advantage)
        value_loss = self._value_loss(value, returns)
        entropy = self.policy_distribution.entropy(policy).mean()

        aug_loss, aug_dict = self._image_augmentation_loss(policy, value, observations)

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
        # 'done' converts into 'not_done' inside '_rollout_to_tensors' method
        observations, actions, rewards, not_done, _ = rollout_t
        policy, value, returns, advantage = self._compute_returns(observations, rewards, not_done)

        loss, result = self.calculate_loss(observations, actions, policy, value, returns, advantage)

        optimization_result = self.optimize_loss(loss)
        result.update(optimization_result)

        return result

    # I want to be able to easily modify data in rollout
    def _train_fn(self, rollout):
        rollout_t = self._rollout_to_tensors(rollout)
        # A2C requires no modifications of rollout

        result = self._main(rollout_t)
        return result
