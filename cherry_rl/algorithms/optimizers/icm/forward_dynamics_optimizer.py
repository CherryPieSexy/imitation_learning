from cherry_rl.algorithms.optimizers.model_optimizer import ModelOptimizer


class ForwardDynamicsModelOptimizer(ModelOptimizer):
    """
    Optimizer for forward dynamics model, i.e. function f: s' ~ p(s'|s, a).
    Uses mse of log_p as loss function.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model.state_distribution_str == 'deterministic':
            self._loss_fn = self._mse_loss
        else:
            self._loss_fn = self._log_p_loss

    def _mse_loss(self, next_observations, model_output, mask):
        mse_loss = 0.5 * (next_observations - model_output) ** 2
        mse_loss = mse_loss.mean(-1)
        mse_loss = self._average_loss(mse_loss, mask)
        return mse_loss

    def _log_p_loss(self, next_observations, model_output, mask):
        log_p = self.model.state_distribution.log_prob(model_output, next_observations)
        log_p_loss = -self._average_loss(log_p, mask)
        return log_p_loss

    def _dynamics_loss(self, data_dict):
        observations = data_dict['obs_emb']
        actions = data_dict['actions']
        mask = data_dict['mask']
        observations, next_observations = observations[:-1], observations[1:]
        model_output = self.model(observations, actions)
        loss = self._loss_fn(next_observations, model_output, mask)
        loss_dict = {'forward_dynamics_loss': loss.item()}
        return loss, loss_dict

    def train(self, data_dict):
        loss, result = self._dynamics_loss(data_dict)
        dynamics_model_grad_norm = self.optimize_loss(loss)
        result.update({'forward_dynamics_model_grad_norm': dynamics_model_grad_norm})
        return result
