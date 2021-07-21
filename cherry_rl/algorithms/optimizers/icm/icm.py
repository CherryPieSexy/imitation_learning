import torch

from cherry_rl.algorithms.optimizers.multi_model_optimizer import MultiModelOptimizer


class ICMOptimizer(MultiModelOptimizer):
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
            actor_critic_optimizer,
            forward_dynamics_optimizer,
            inverse_dynamics_optimizer,
            dynamics_encoder_factory=None,
            extrinsic_reward_weight=1.0,
            intrinsic_reward_weight=1.0,
            clip_grad=0.5,
            warm_up_steps=1000
    ):
        self._actor_critic_optimizer = actor_critic_optimizer
        self._gamma = self._actor_critic_optimizer.gamma
        self._forward_dynamics_optimizer = forward_dynamics_optimizer
        self._inverse_dynamics_optimizer = inverse_dynamics_optimizer

        self._dynamics_encoder = dynamics_encoder_factory()
        dynamics_encoder_parameters = list(self._dynamics_encoder.parameters())
        if dynamics_encoder_parameters:
            self._dynamics_encoder_optimizer = torch.optim.Adam(dynamics_encoder_parameters, lr=1e-3)
            self._dynamics_encoder_optimizer.zero_grad()
        else:
            self._dynamics_encoder_optimizer = None
        self._clip_grad = clip_grad
        self._warm_up_steps = warm_up_steps
        self._step = 0

        self._extrinsic_reward_weight = extrinsic_reward_weight
        self._intrinsic_reward_weight = intrinsic_reward_weight

        self._discounted_intrinsic_return = 0
        self._alive_envs = 1

    def _change_rewards(self, rollout_data_dict):
        not_done = 1.0 - rollout_data_dict['is_done']

        observations = rollout_data_dict['obs_emb']
        actions = rollout_data_dict['actions']

        with torch.no_grad():
            fdm_prediction = self._forward_dynamics_optimizer.model.sample(
                observations[:-1], actions, deterministic=True
            )

        extrinsic_rewards = rollout_data_dict['rewards']
        intrinsic_rewards = 0.5 * (observations[1:] - fdm_prediction) ** 2
        intrinsic_rewards = intrinsic_rewards.mean(-1).unsqueeze(-1)
        new_rewards = \
            self._extrinsic_reward_weight * extrinsic_rewards + self._intrinsic_reward_weight * intrinsic_rewards
        rollout_data_dict['rewards'] = new_rewards

        intrinsic_returns = torch.zeros_like(intrinsic_rewards)  # (time, batch, 1)
        for t in range(intrinsic_returns.size(0)):
            # TODO: there is an 'shape' problem,
            #  it is not clear of which shape discounted return should be
            self._discounted_intrinsic_return = \
                self._gamma * self._alive_envs * self._discounted_intrinsic_return + intrinsic_rewards[t]
            self._alive_envs = not_done[t]
            intrinsic_returns[t] = self._discounted_intrinsic_return
        extrinsic_returns = rollout_data_dict['returns']
        rollout_data_dict['returns'] =\
            self._extrinsic_reward_weight * extrinsic_returns + self._intrinsic_reward_weight * intrinsic_returns
        return rollout_data_dict, intrinsic_rewards.mean().item()

    def _optimize_encoder(self):
        grad_norm = None
        if self._dynamics_encoder_optimizer is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self._dynamics_encoder.parameters(), self._clip_grad)
            self._dynamics_encoder_optimizer.step()
            self._dynamics_encoder_optimizer.zero_grad()
        return grad_norm

    def train(self, rollout_data_dict):
        rollout_data_dict = self._actor_critic_optimizer.model.t(rollout_data_dict)
        rollout_data_dict['obs_emb'] = self._dynamics_encoder(rollout_data_dict['observations'])

        # optimize IDM, FDM and dynamics encoder on observation embeddings
        idm_optimization_result = self._inverse_dynamics_optimizer.train(rollout_data_dict)
        rollout_data_dict['obs_emb'] = rollout_data_dict['obs_emb'].detach()
        fdm_optimization_result = self._forward_dynamics_optimizer.train(rollout_data_dict)
        encoder_grad_norm = self._optimize_encoder()

        icm_rewards_mean = 0.0
        ac_optimization_result, ac_time = dict(), dict()
        if self._step > self._warm_up_steps:
            # change rewards and optimize RL
            rollout_data_dict, icm_rewards_mean = self._change_rewards(rollout_data_dict)
            ac_optimization_result = self._actor_critic_optimizer.train(rollout_data_dict)

            ac_time = dict()
            if isinstance(ac_optimization_result, tuple):
                ac_optimization_result, ac_time = ac_optimization_result

        result = ac_optimization_result
        result.update(idm_optimization_result)
        result.update(fdm_optimization_result)
        result.update({'icm_reward': icm_rewards_mean})
        if encoder_grad_norm is not None:
            result.update({'dynamics_encoder_grad_norm': encoder_grad_norm})

        self._step += 1
        return result, ac_time

    def state_dict(self):
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

    def save(self, filename):
        torch.save(self.state_dict(), filename)
