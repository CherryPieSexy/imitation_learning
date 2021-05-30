import torch

from cherry_rl.utils.utils import time_it
from cherry_rl.algorithms.optimizers.multi_model_optimizer import MultiModelOptimizer


class BCO(MultiModelOptimizer):
    """Behavioral Cloning from Observation.

    https://arxiv.org/abs/1805.01954
    Works in pair with 'PolicyOptimizer' and 'InverseDynamicsOptimizer'.
    Main idea of the algorithm:
        1) collect rollout data by an agent
        2) train inverse dynamics model (idm) on gathered data
        3) apply trained idm to demo data to infer expert actions
        4) train policy to do same actions as expert did

    Possible improvement: use several mini-batch updates per data-piece, like in PPO.
    """
    def __init__(
            self,
            policy_optimizer,
            inverse_dynamics_optimizer,
            demo_buffer,
            use_idm_grad=False,
            obs_encoder_lr=3e-4,
            clip_obs_encoder_grad=0.1
    ):
        # This 'actor-critic' optimizer must be an instance of 'PolicyOptimizer' class.
        self._actor_critic_optimizer = policy_optimizer
        self._inverse_dynamics_optimizer = inverse_dynamics_optimizer
        self._demo_buffer = demo_buffer

        self._idm_deterministic = inverse_dynamics_optimizer.model.action_distribution_str == 'deterministic'

        self._use_idm_grad = use_idm_grad
        if self._use_idm_grad:
            self._init_obs_encoder_optimizer(
                policy_optimizer, obs_encoder_lr, clip_obs_encoder_grad
            )

    def update_obs_normalizer(self, data_dict, drop_last=True):
        if self._actor_critic_optimizer.model.obs_normalizer is not None:
            # Select all observations except the last to correctly sum with mask.
            observation_t = data_dict['observations']
            if drop_last:
                observation_t = observation_t[: -1]
            mask = data_dict['mask']
            self._actor_critic_optimizer.model.obs_normalizer.update(observation_t, mask)

    def _idm_and_demo_action_discrepancy(self, idm_prediction, demo_actions):
        if self._idm_deterministic:
            idm_action = idm_prediction
        else:
            idm_action, _ = self._inverse_dynamics_optimizer.model.action_distribution.sample(
                idm_prediction, deterministic=True
            )
        diff = 0.5 * (idm_action - demo_actions) ** 2
        diff = diff.mean()
        return diff

    def _obs_embedding(self, data_dict, **kwargs):
        return super()._obs_embedding(data_dict, self._use_idm_grad, **kwargs)

    def _train_fn(self, rollout_data_dict, demo_data_dict):
        rollout_data_dict['obs_emb'] = self._obs_embedding(rollout_data_dict)
        demo_data_dict['obs_emb'] = self._obs_embedding(demo_data_dict)
        demo_data_dict['next_obs_emb'] = self._obs_embedding(demo_data_dict, obs_key='next_observations')

        idm_train_result = self._inverse_dynamics_optimizer.train(rollout_data_dict)
        with torch.no_grad():
            idm_prediction = self._inverse_dynamics_optimizer.model(
                demo_data_dict['obs_emb'], demo_data_dict['next_obs_emb']
            )

        encoder_grad_norm = None
        if self._use_idm_grad:
            encoder_grad_norm = self._train_encoder()

        # Policy model will compute embeddings by itself.
        policy_train_result = self._actor_critic_optimizer.train(
            demo_data_dict['observations'], idm_prediction,
            demo_data_dict['mask'], self._idm_deterministic
        )

        result = idm_train_result
        result.update(policy_train_result)
        if 'actions' in demo_data_dict:
            idm_demo_discrepancy = self._idm_and_demo_action_discrepancy(
                idm_prediction, demo_data_dict['actions']
            )
            result.update({'idm_and_demo_discrepancy': idm_demo_discrepancy})
        if self._use_idm_grad:
            result.update({'encoder.idm_grad_norm': encoder_grad_norm})
        return result

    def train(self, rollout_data_dict):
        demo_data_dict = self._demo_buffer.sample()
        demo_data_dict = self._actor_critic_optimizer.model.t(demo_data_dict)
        train_fn_result, train_fn_time = time_it(self._train_fn)(rollout_data_dict, demo_data_dict)
        self.update_obs_normalizer(rollout_data_dict)
        self.update_obs_normalizer(demo_data_dict, drop_last=False)
        time_log = {'train_on_rollout': train_fn_time}
        return train_fn_result, time_log

    def state_dict(self):
        state_dict = {
            'ac_model': self._actor_critic_optimizer.model.state_dict(),
            'ac_optimizer': self._actor_critic_optimizer.optimizer.state_dict(),
            'idm_model': self._inverse_dynamics_optimizer.model.state_dict(),
            'idm_optimizer': self._inverse_dynamics_optimizer.optimizer.state_dict(),
        }
        return state_dict

    def save(self, filename):
        torch.save(self.state_dict(), filename)
