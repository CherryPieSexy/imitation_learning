from algorithms.agents.policy_gradient import PolicyGradient


class A2C(PolicyGradient):
    """
    can act (with grad, it may be useful) and calculate loss on rollout.
    This class defines A2C-specific parts, such as
    loss-function and how to call it
    """

    def _policy_loss(self, policy, actions, advantages):
        log_pi_for_actions = self.policy_distribution.log_prob(policy, actions)
        policy_loss = log_pi_for_actions * advantages.detach()
        return policy_loss.mean()

    def _main(self, observations, policy, values, actions, returns, advantages):
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        policy_loss = self._policy_loss(policy, actions, advantages)
        entropy = self.policy_distribution.entropy(policy).mean()
        loss = value_loss - policy_loss - self.entropy * entropy
        if self.image_augmentation_alpha > 0.0:
            (policy_div, value_div), img_aug_time = self._augmentation_loss(
                policy.detach(), values.detach(), observations
            )
            loss += self.image_augmentation_alpha * (policy_div + value_div)
            upd = {
                'policy_div': policy_div.item(),
                'value_div': value_div.item(),
                'img_aug_time': img_aug_time
            }

        grad_norm = self._optimize_loss(loss)

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

    def _train_fn(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done, log_probs),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
               I want to store 'log_probs' inside rollout
               because online policy (i.e. the policy gathered rollout)
               may not be the trained policy
        :return: dict of loss values, gradient norm
        """
        # 'done' converts into 'not_done' inside '_rollout_to_tensors' method
        observations, actions, rewards, not_done, _ = self._rollout_to_tensors(rollout)

        policy, values, returns, advantage = self._compute_returns(observations, rewards, not_done)

        result_log = self._main(observations, policy, values, actions, returns, advantage)
        time_log = dict()
        if self.image_augmentation_alpha > 0.0:
            time_log['img_aug'] = result_log.pop('img_aug_time')

        return result_log, time_log
