from typing import Dict, Optional

import torch

from cherry_rl.algorithms.optimizers.model_optimizer import ModelOptimizer


class InverseDynamicsOptimizer(ModelOptimizer):
    """
    Optimizer for inverse dynamics model, i.e. function f: a ~ f(s, s').
    Optimizes dynamics using log-prob as loss.
    """
    def loss(
            self,
            rollout_data_dict: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        # suppose that observations are time-ordered.
        observations = rollout_data_dict['obs_emb']
        current_observations, next_observations = observations[:-1], observations[1:]
        actions = rollout_data_dict['actions']

        model_prediction = self.model(current_observations, next_observations)
        actions_log_prob = self.model.action_distribution.log_prob(
            model_prediction, actions
        )
        loss = -actions_log_prob

        return loss

    def train(
            self,
            data_dict: Dict[str, Optional[torch.Tensor]],
            retain_graph: bool = False
    ) -> Dict[str, float]:
        """
        In ICM algorithm inverse and forward dynamics share embedding,
        so it may be necessary to retain graph during training encoder.

        :param data_dict:
        :param retain_graph:
        :return:
        """
        loss = self.loss(data_dict)
        loss = self._average_loss(loss, data_dict['mask'])
        result = {'inverse_dynamics_loss': loss.item()}

        inverse_dynamics_model_grad_norm = self.optimize_loss(loss, retain_graph)
        result.update({'inverse_dynamics_model_grad_norm': inverse_dynamics_model_grad_norm})
        return result
