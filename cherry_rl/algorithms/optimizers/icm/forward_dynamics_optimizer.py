from typing import Optional, Dict, Union, Tuple

import torch

from cherry_rl.algorithms.optimizers.model_optimizer import ModelOptimizer


class ForwardDynamicsModelOptimizer(ModelOptimizer):
    """
    Optimizer for forward dynamics model, i.e. function f: s' ~ p(s'|s, a).
    Uses log-prob as loss function.
    """
    def loss(
            self,
            data_dict: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        observations = data_dict['obs_emb']
        actions = data_dict['actions']
        observations, next_observations = observations[:-1], observations[1:].detach()
        model_output = self.model(observations, actions)
        log_p = self.model.state_distribution.log_prob(model_output, next_observations)
        loss = -log_p
        return loss

    def train(
            self,
            data_dict: Dict[str, Optional[torch.Tensor]],
            return_averaged: bool = True
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], torch.Tensor]]:
        loss = self.loss(data_dict)
        loss_averaged = self._average_loss(loss, data_dict['mask'])
        result = {'forward_dynamics_loss': loss_averaged.item()}

        dynamics_model_grad_norm = self.optimize_loss(loss_averaged)
        result.update({'forward_dynamics_model_grad_norm': dynamics_model_grad_norm})
        return result if return_averaged else result, loss
