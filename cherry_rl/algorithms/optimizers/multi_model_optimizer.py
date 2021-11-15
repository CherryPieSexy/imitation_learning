from typing import Dict, Optional

import torch


class MultiModelOptimizer:
    """
    Multi-model optimizer is useful when there is an encoder inside actor-critic,
    because encoder's optimization and gradients controlled by actor-critic optimizer
    and sometimes we want to use gradients from different models (discriminator or dynamics).

    This class creates additional encoder-optimizer and provide method to train encoder
    independent of actor-critic using gradients accumulated from other models.
    """
    _actor_critic_optimizer = None
    _obs_encoder = None
    _encoder_optimizer = None
    _clip_obs_encoder_grad = None

    def _init_obs_encoder_optimizer(
            self, ac_optimizer, lr, clip_grad
    ):
        self._obs_encoder = ac_optimizer.model.obs_encoder
        self._encoder_optimizer = torch.optim.Adam(
            self._obs_encoder.parameters(), lr
        )
        self._clip_obs_encoder_grad = clip_grad

    def _obs_embedding(
            self,
            data_dict: Dict[str, Optional[torch.Tensor]],
            memory: Optional = None,
            with_grad: bool = True,
            obs_key: str = 'observations'
    ) -> torch.Tensor:
        obs = data_dict[obs_key]
        if with_grad:
            emb, memory = self._actor_critic_optimizer.model.preprocess_observation(obs, memory)
        else:
            with torch.no_grad():
                emb, memory = self._actor_critic_optimizer.model.preprocess_observation(obs, memory)
        return emb, memory

    def _train_encoder(self) -> float:
        # encoder should already have gradients
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._obs_encoder.parameters(), self._clip_obs_encoder_grad
        )
        self._encoder_optimizer.step()
        self._encoder_optimizer.zero_grad()
        return encoder_grad_norm
