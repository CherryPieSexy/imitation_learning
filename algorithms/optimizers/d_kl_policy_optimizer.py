from algorithms.kl_divergence import kl_divergence
from algorithms.optimizers.model_optimizer import ModelOptimizer


class PolicyOptimizer(ModelOptimizer):
    """
    Simple policy-model optimizer.

    It used by 'Behavioral Cloning from Observations'.
    Accepts distribution / action as target
    and minimize D_KL / MSE between it and policy.
    """
    def __init__(
            self,
            policy_model,
            learning_rate=3e-4,
            clip_grad=0.5,
            entropy=0.0
    ):
        super().__init__(policy_model, learning_rate, clip_grad)
        self.entropy = entropy

    def _policy_loss(
            self,
            observations, target_policy, mask,
            target_deterministic
    ):
        policy, _, _ = self.model(observations, None)
        entropy = self.model.pi_distribution.entropy(policy)
        if target_deterministic:
            policy, _ = self.model.pi_distribution.sample(policy, deterministic=True)
            policy_loss = 0.5 * (policy - target_policy) ** 2
            policy_loss = policy_loss.mean(-1)
        else:
            # We want to move policy towards idm prediction.
            # I believe it corresponds to minimizing D_KL(idm, policy).
            policy_loss = kl_divergence(
                self.model.pi_distribution_str, target_policy, policy
            )

        policy_loss = self._average_loss(policy_loss, mask)
        entropy_loss = self._average_loss(entropy, mask)
        loss = policy_loss - self.entropy * entropy_loss
        loss_dict = {
            'mse_policy_loss' if target_deterministic else 'd_kl_loss': policy_loss.item(),
            'entropy': entropy_loss.item(),
            'policy_loss': loss.item()
        }
        return loss, loss_dict

    def train(self, observations, target_policy, mask, target_deterministic=False):
        loss, result = self._policy_loss(observations, target_policy, mask, target_deterministic)
        policy_model_grad_norm = self.optimize_loss(loss)
        result.update({'policy_model_grad_norm': policy_model_grad_norm})
        return result
