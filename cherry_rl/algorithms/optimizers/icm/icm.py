from typing import Optional, Callable, Any, Dict, Tuple

import torch


class ICMOptimizer:
    """
    Intrinsic Curiosity Module.
    https://arxiv.org/abs/1705.05363

    Jointly optimizes forward and inverse dynamics models,
    computes intrinsic reward, sum it with extrinsic and send result into an RL optimizer.
    Can be treated as RL algo for combining with different algorithms,
    for example with GAIL.

    Inverse dynamics trained to disentangle part of observation
    on which agent have influence, forward dynamics used for measuring
    agent 'surprise' on observations, if observation is 'surprising'
    then it is beneficial to visit it again.
    """
    def __init__(
            self,
            make_ac_optimizer: Callable[[], Any],
            make_forward_dynamics_optimizer: Callable[[], Any],
            make_inverse_dynamics_optimizer: Callable[[], Any],
            dynamics_encoder_factory: Callable[[], Any],
            extrinsic_reward_weight: float = 1.0,
            intrinsic_reward_weight: float = 1.0,
            beta: float = 0.2,
            encoder_lr: float = 1e-3,
            clip_grad: float = 0.5,
            warm_up_steps: int = 1000
    ):
        self._actor_critic_optimizer = make_ac_optimizer()
        self._gamma = self._actor_critic_optimizer.gamma
        self._forward_dynamics_optimizer = make_forward_dynamics_optimizer()
        self._inverse_dynamics_optimizer = make_inverse_dynamics_optimizer()

        self._dynamics_encoder = dynamics_encoder_factory()
        self._dynamics_encoder_optimizer = torch.optim.Adam(self._dynamics_encoder.parameters(), lr=encoder_lr)
        self._clip_grad = clip_grad
        self._warm_up_steps = warm_up_steps
        self._step = 0

        self._extrinsic_reward_weight = extrinsic_reward_weight
        self._intrinsic_reward_weight = intrinsic_reward_weight
        self._beta = beta

        self._discounted_intrinsic_return = 0
        self._alive_envs = 1

    @staticmethod
    def _average_loss(
            loss: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(loss)
        return (mask * loss).sum() / mask.sum()

    def _train_dynamics(
            self,
            rollout_data_dict: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # I have to do training step (i.e. zero grad, compute loss, optimizer step) manually
        # because IDM & FDM share observation embedding
        rollout_data_dict['obs_emb'] = self._dynamics_encoder(rollout_data_dict['observations'])

        self._inverse_dynamics_optimizer.optimizer.zero_grad()
        self._forward_dynamics_optimizer.optimizer.zero_grad()
        self._dynamics_encoder_optimizer.zero_grad()

        # losses from dynamics are tensors of shape [Time, Batch]
        idm_loss = self._inverse_dynamics_optimizer.loss(rollout_data_dict)
        fdm_loss = self._forward_dynamics_optimizer.loss(rollout_data_dict)
        icm_rewards = fdm_loss.detach()
        idm_loss = self._average_loss(idm_loss, rollout_data_dict['mask'])
        fdm_loss = self._average_loss(idm_loss, rollout_data_dict['mask'])

        dynamics_loss = (1.0 - self._beta) * idm_loss + self._beta * fdm_loss
        dynamics_loss.backward()
        idm_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._inverse_dynamics_optimizer.model.parameters(), self._clip_grad
        )
        fdm_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._forward_dynamics_optimizer.model.parameters(), self._clip_grad
        )
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(self._dynamics_encoder.parameters(), self._clip_grad)
        self._inverse_dynamics_optimizer.step()
        self._forward_dynamics_optimizer.step()
        self._dynamics_encoder_optimizer.step()

        result = {
            'inverse_dynamics_loss': idm_loss.item(),
            'forward_dynamics_loss': fdm_loss.item(),
            'icm_rewards': icm_rewards.mean().item(),

            'inverse_dynamics_model_grad_norm': idm_grad_norm,
            'forward_dynamics_model_grad_norm': fdm_grad_norm,
            'dynamics_encoder_grad_norm': encoder_grad_norm,
        }

        return icm_rewards, result

    def _change_rewards(
            self,
            icm_rewards: torch.Tensor,
            rollout_data_dict: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        done = rollout_data_dict['is_done']
        rollout_data_dict['is_done'] = torch.cat([done, torch.zeros_like(done)], dim=-1)

        extrinsic_rewards = rollout_data_dict['rewards']
        intrinsic_rewards = icm_rewards.mean(-1).unsqueeze(-1)

        new_rewards = torch.cat([
            self._extrinsic_reward_weight * extrinsic_rewards,
            self._intrinsic_reward_weight * intrinsic_rewards
        ], dim=-1)

        # TODO: do I need to change returns here? How can I change it?

        rollout_data_dict['rewards'] = new_rewards
        return rollout_data_dict

    def train(
            self,
            rollout_data_dict: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[Dict[str, float], Dict[Optional[str], Optional[float]]]:
        """Training with non-stopped gradients"""
        rollout_data_dict = self._actor_critic_optimizer.model.t(rollout_data_dict)
        icm_rewards, result = self._train_dynamics(rollout_data_dict)

        ac_optimization_result, ac_time = dict(), dict()
        if self._step > self._warm_up_steps:
            rollout_data_dict = self._change_rewards(icm_rewards, rollout_data_dict)
            ac_optimization_result = self._actor_critic_optimizer.train(rollout_data_dict)

            ac_time = dict()
            if isinstance(ac_optimization_result, tuple):
                ac_optimization_result, ac_time = ac_optimization_result

        result.update(ac_optimization_result)
        self._step += 1
        return result, ac_time

    def state_dict(self) -> Dict[str, dict]:
        state_dict = {
            'ac_model': self._actor_critic_optimizer.model.state_dict(),
            'ac_optimizer': self._actor_critic_optimizer.optimizer.state_dict(),
            'forward_dynamics_model': self._forward_dynamics_optimizer.model.state_dict(),
            'forward_dynamics_model_optimizer': self._forward_dynamics_optimizer.optimizer.state_dict(),
            'inverse_dynamics_model': self._inverse_dynamics_optimizer.model.state_dict(),
            'inverse_dynamics_model_optimizer': self._inverse_dynamics_optimizer.optimizer.state_dict(),
        }
        if self._dynamics_encoder_optimizer is not None:
            state_dict.update({
                'dynamics_feature_extractor_model': self._dynamics_encoder.state_dict(),
                'dynamics_feature_extractor_optimizer': self._dynamics_encoder_optimizer.state_dict(),
            })
        return state_dict

    def save(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)
