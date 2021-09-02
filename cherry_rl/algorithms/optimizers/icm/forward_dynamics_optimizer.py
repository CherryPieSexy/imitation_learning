from typing import Optional, Dict

import torch

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

    @staticmethod
    def _mse_loss(
            model_output: torch.Tensor,
            next_observations: torch.Tensor
    ) -> torch.Tensor:
        mse_loss = 0.5 * (model_output - next_observations) ** 2
        loss = mse_loss.mean(-1)
        return loss

    def _log_p_loss(
            self,
            model_output: torch.Tensor,
            next_observations: torch.Tensor
    ) -> torch.Tensor:
        log_p = self.model.state_distribution.log_prob(model_output, next_observations)
        loss = -log_p
        return loss

    def loss(
            self,
            data_dict: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        observations = data_dict['obs_emb']
        actions = data_dict['actions']
        observations, next_observations = observations[:-1], observations[1:].detach()
        model_output = self.model(observations, actions)
        loss = self._loss_fn(model_output, next_observations)
        return loss

    def train(
            self,
            data_dict: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, float]:
        loss = self.loss(data_dict)
        loss = self._average_loss(loss, data_dict['mask'])
        result = {'forward_dynamics_loss': loss.item()}

        dynamics_model_grad_norm = self.optimize_loss(loss)
        result.update({'forward_dynamics_model_grad_norm': dynamics_model_grad_norm})
        return result
