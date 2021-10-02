from typing import Callable

import torch
from torch.nn.functional import softplus

from cherry_rl.algorithms.optimizers.model_optimizer import ModelOptimizer


# logit - lambda x: x
# sigmoid - torch.sigmoid
def log_sigmoid(x):
    # (-inf, 0)
    return torch.log(torch.sigmoid(x))


def neg_log_sigmoid(x):
    # (0, +inf)
    return -torch.log(1.0 - torch.sigmoid(x))


def log_sum(x):
    return torch.log(torch.sigmoid(x)) - torch.log(1.0 - torch.sigmoid(x))


class DiscriminatorOptimizer(ModelOptimizer):
    """
    Optimizer for observation/rollout discriminator. Useful for GAIL.

    Selecting correct reward type is important for learning:
        in my experiments, log_sigmoid and neg_log_sigmoid
              works better than logit and sigmoid.
        log_sigmoid is always negative, it force agent to end episode asap;
        neg_log_sigmoid is always positive, it force agent to stay alive.
    """
    def __init__(
            self,
            model,
            learning_rate: float,
            clip_grad: float,
            reward_fn: Callable = lambda x: x,
            discriminator_with_actions: bool = False
    ):
        super().__init__(model, learning_rate, clip_grad)
        self._reward_fn = reward_fn
        self._discriminator_with_actions = discriminator_with_actions

    def _calculate_logits(self, data_dict):
        observations = data_dict['obs_emb']
        if self._discriminator_with_actions:
            actions = data_dict['actions']
            logits = self.model(observations, actions)
        else:
            logits = self.model(observations)
        return logits

    def predict_reward(self, data_dict):
        logits = self._calculate_logits(data_dict)
        reward = self._reward_fn(logits)
        return reward

    def _discriminator_loss(self, rollout_data_dict, demo_data_dict):
        rollout_logits = self._calculate_logits(rollout_data_dict).squeeze(-1)
        demo_logits = self._calculate_logits(demo_data_dict).squeeze(-1)
        # TODO: what about sequential discriminator and loss averaging?
        # TODO: understand and explain this 'softplus' loss in class docstring.
        rollout_loss = self._average_loss(softplus(rollout_logits), rollout_data_dict['mask'])
        demo_loss = self._average_loss(softplus(-demo_logits), demo_data_dict['mask'])
        loss = rollout_loss + demo_loss

        loss_dict = {
            'discriminator_on_roll': rollout_logits.mean().item(),
            'discriminator_on_demo': demo_logits.mean().item(),
            'discriminator_loss': loss.item()
        }

        return loss, loss_dict

    def train(self, rollout_data_dict, demo_data_dict):
        loss, result = self._discriminator_loss(rollout_data_dict, demo_data_dict)
        discriminator_grad_norm = self.optimize_loss(loss)
        result.update({'discriminator_grad_norm': discriminator_grad_norm})
        return result
