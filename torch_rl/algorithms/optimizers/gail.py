import torch

from torch_rl.algorithms.optimizers.multi_model_optimizer import MultiModelOptimizer


class GAIL(MultiModelOptimizer):
    """
    Generative Adversarial Imitation Learning.
    https://arxiv.org/abs/1606.03476

    Main idea of the algorithm:
        1) collect rollout data by an agent
        2) train discriminator to distinguish between demo data and rollout data
        3) train agent to fool discriminator by any RL algorithm
    """
    def __init__(
            self,
            actor_critic_optimizer,
            discriminator_optimizer,
            demo_buffer,
            use_discriminator_grad=False,
            obs_encoder_lr=3e-4,
            clip_obs_encoder_grad=0.1
    ):
        self._actor_critic_optimizer = actor_critic_optimizer
        self._gamma = self._actor_critic_optimizer.gamma
        self._discriminator_optimizer = discriminator_optimizer
        self._demo_buffer = demo_buffer

        self._use_discriminator_grad = use_discriminator_grad

        if self._use_discriminator_grad:
            self._init_obs_encoder_optimizer(
                actor_critic_optimizer, obs_encoder_lr, clip_obs_encoder_grad
            )

        self._discounted_gail_return = 0
        self._alive_envs = 1

    def _change_rewards(self, rollout_data_dict):
        not_done = 1.0 - rollout_data_dict['is_done']
        with torch.no_grad():
            gail_rewards = self._discriminator_optimizer.predict_reward(rollout_data_dict)
        rollout_data_dict['rewards'] = gail_rewards
        gail_returns = torch.zeros_like(gail_rewards)
        for t in range(gail_returns.size(0)):
            self._discounted_gail_return = \
                self._gamma * self._alive_envs * self._discounted_gail_return + gail_rewards[t]
            self._alive_envs = not_done[t]
            gail_returns[t] = self._discounted_gail_return
        rollout_data_dict['returns'] = gail_returns
        return rollout_data_dict, gail_rewards.mean().item()

    def _obs_embedding(self, data_dict, **kwargs):
        return super()._obs_embedding(data_dict, self._use_discriminator_grad)

    def train(self, rollout_data_dict):
        demo_data_dict = self._demo_buffer.sample()
        demo_data_dict = self._actor_critic_optimizer.model.t(demo_data_dict)
        # we don't need last observation from rollout, just drop it.
        rollout_data_dict['obs_emb'] = self._obs_embedding(rollout_data_dict)[:-1]
        demo_data_dict['obs_emb'] = self._obs_embedding(demo_data_dict)

        discriminator_optimization_result = self._discriminator_optimizer.train(
            rollout_data_dict, demo_data_dict
        )

        encoder_grad_norm = None
        if self._use_discriminator_grad:
            encoder_grad_norm = self._train_encoder()

        rollout_data_dict, gail_rewards_mean = self._change_rewards(rollout_data_dict)
        ac_optimization_result = self._actor_critic_optimizer.train(rollout_data_dict)

        if self._actor_critic_optimizer.model.obs_normalizer is not None:
            self._actor_critic_optimizer.model.obs_normalizer.update(
                demo_data_dict['observations'], demo_data_dict['mask']
            )

        ac_time = dict()
        if isinstance(ac_optimization_result, tuple):
            ac_optimization_result, ac_time = ac_optimization_result

        result = ac_optimization_result
        result.update(discriminator_optimization_result)
        result.update({'gail_rewards': gail_rewards_mean})
        if self._use_discriminator_grad:
            result.update({'encoder.gail_grad_norm': encoder_grad_norm})
        return result, ac_time

    def state_dict(self):
        state_dict = {
            'ac_model': self._actor_critic_optimizer.model.state_dict(),
            'ac_optimizer': self._actor_critic_optimizer.optimizer.state_dict(),
            'discriminator_model': self._discriminator_optimizer.model.state_dict(),
            'discriminator_optimizer': self._discriminator_optimizer.optimizer.state_dict(),
        }
        if self._encoder_optimizer is not None:
            state_dict.update({'encoder_optimizer': self._encoder_optimizer.state_dict()})
        return state_dict

    def save(self, filename):
        torch.save(self.state_dict(), filename)
