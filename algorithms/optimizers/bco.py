import torch

from utils.utils import time_it


class BCO:
    """Behavioral Cloning from Observation.

    https://arxiv.org/abs/1805.01954
    Main idea of the algorithm:
        1) collect rollout data by an agent
        2) train inverse dynamics model (idm) on gathered data
        3) apply trained idm to demo data to infer expert actions
        4) train policy to do same actions as expert did
    """
    def __init__(
            self,
            policy_optimizer,
            inverse_dynamics_optimizer,
            demo_buffer
    ):
        self._policy_optimizer = policy_optimizer
        self._inverse_dynamics_optimizer = inverse_dynamics_optimizer
        self._demo_buffer = demo_buffer

        self._idm_deterministic = inverse_dynamics_optimizer.model.action_distribution_str == 'deterministic'

    def update_obs_normalizer(self, data_dict, drop_last=True):
        if self._policy_optimizer.model.obs_normalizer is not None:
            # select all observations except the last to correctly sum with mask.
            observation_t = data_dict['observations']
            if drop_last:
                observation_t = observation_t[: -1]
            mask = data_dict['mask']
            self._policy_optimizer.model.obs_normalizer.update(observation_t, mask)

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

    def _compute_embeddings(self, rollout_data_dict, demo_data_dict):
        with torch.no_grad():
            rollout_obs_emb, _ = self._policy_optimizer.model.preprocess_observation(
                rollout_data_dict['observations'], None
            )
            demo_obs_emb, _ = self._policy_optimizer.model.preprocess_observation(
                demo_data_dict['observations'], None
            )

        rollout_data_dict['obs_emb'] = rollout_obs_emb
        demo_data_dict['obs_emb'] = demo_obs_emb
        return rollout_data_dict, demo_data_dict

    def _train_fn(self, rollout_data_dict, demo_data_dict):
        rollout_data_dict, demo_data_dict = self._compute_embeddings(
            rollout_data_dict, demo_data_dict
        )

        idm_train_result = self._inverse_dynamics_optimizer.train(rollout_data_dict)
        with torch.no_grad():
            idm_prediction = self._inverse_dynamics_optimizer.model(
                demo_data_dict['observations'], demo_data_dict['next_observations']
            )

        policy_train_result = self._policy_optimizer.train(
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
        return result

    def train(self, rollout_data_dict):
        demo_data_dict = self._demo_buffer.sample()
        demo_data_dict = self._policy_optimizer.model.t(demo_data_dict)
        train_fn_result, train_fn_time = time_it(self._train_fn)(rollout_data_dict, demo_data_dict)
        self.update_obs_normalizer(rollout_data_dict)
        self.update_obs_normalizer(demo_data_dict, drop_last=False)
        time_log = {'train_on_rollout': train_fn_time}
        return train_fn_result, time_log

    def state_dict(self):
        state_dict = {
            'ac_model': self._policy_optimizer.model.state_dict(),
            'ac_optimizer': self._policy_optimizer.optimizer.state_dict(),
            'idm_model': self._inverse_dynamics_optimizer.model.state_dict(),
            'idm_optimizer': self._inverse_dynamics_optimizer.optimizer.state_dict(),
        }
        return state_dict

    def save(self, filename):
        torch.save(self.state_dict(), filename)
