import torch
from algorithms.optimizers.actor_critic_optimizer import ActorCriticOptimizer


class A2C(ActorCriticOptimizer):
    def _policy_loss(self, policy, actions, advantage):
        if self.normalize_adv:
            advantage = advantage - advantage.mean() / (advantage.std() + 1e-8)
        log_pi_for_actions = self.model.pi_distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantage.detach()
        return policy_loss

    @staticmethod
    def _value_loss(value, returns):
        difference_2 = (value - returns) ** 2
        value_loss = 0.5 * (difference_2.mean(-1))
        return value_loss

    def calculate_loss(self, data_dict, policy, value):
        actions = data_dict['actions']
        returns = data_dict['returns']
        advantage = data_dict['advantage']
        mask = data_dict['mask']

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
    def _main(self, data_dict):
        data_dict = {
            k: (v.to(torch.float32) if v is not None else None)
            for k, v in data_dict.items()
        }

        policy, value, returns, advantage = self.returns_estimator.policy_value_returns_adv(
            self.model, data_dict
        )
        data_dict['returns'] = returns
        data_dict['advantage'] = advantage

        if self.model.value_normalizer is not None:
            self.model.value_normalizer.update(returns, data_dict['mask'])

        # policy, value = policy[:-1], value[:-1]
        loss, result = self.calculate_loss(
            data_dict, policy, value
        )

        gradient_norms = self.optimize_loss(loss)
        result.update(gradient_norms)

        return result

    # I want to be able to easily modify data in rollout
    def _train_fn(self, data_dict):
        # A2C requires no modifications of rollout
        result = self._main(data_dict)
        return result
