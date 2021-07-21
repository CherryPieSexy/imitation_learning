from cherry_rl.algorithms.optimizers.model_optimizer import ModelOptimizer


class InverseDynamicsOptimizer(ModelOptimizer):
    """
    Optimizer for inverse dynamics model, i.e. function f: a ~ f(s, s').
    Can optimize deterministic dynamics (use mse as loss) and stochastic (use log-prob as loss).
    """
    def __init__(
            self,
            model,
            learning_rate=3e-4,
            clip_grad=0.5
    ):
        super().__init__(model, learning_rate, clip_grad)

    def _id_loss(self, model_prediction, actions):
        # id = inverse dynamics
        if self.model.action_distribution_str == 'deterministic':
            loss = 0.5 * (model_prediction - actions) ** 2
            loss = loss.mean(-1)
        else:
            actions_log_prob = self.model.action_distribution.log_prob(
                model_prediction, actions
            )
            loss = -actions_log_prob
        return loss

    def _inverse_dynamics_loss(self, rollout_data_dict):
        # suppose that observations are time-ordered.
        observations = rollout_data_dict['obs_emb']
        current_observations, next_observations = observations[:-1], observations[1:]
        actions = rollout_data_dict['actions']
        mask = rollout_data_dict['mask']

        model_prediction = self.model(current_observations, next_observations)
        loss = self._id_loss(model_prediction, actions)

        loss = self._average_loss(loss, mask)
        loss_dict = {'inverse_dynamics_loss': loss.item()}
        return loss, loss_dict

    def train(self, data_dict):
        loss, result = self._inverse_dynamics_loss(data_dict)
        inverse_dynamics_model_grad_norm = self.optimize_loss(loss)
        result.update({'inverse_dynamics_model_grad_norm': inverse_dynamics_model_grad_norm})
        return result
