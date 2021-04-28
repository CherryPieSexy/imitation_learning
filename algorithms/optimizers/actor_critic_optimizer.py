import torch

from utils.utils import time_it
from algorithms.optimizers.model_optimizer import ModelOptimizer
from algorithms.returns_estimator import ReturnsEstimator


class ActorCriticOptimizer(ModelOptimizer):
    # inherit a2c and ppo from it with definition of train_fn or something
    def __init__(
            self,
            model,
            learning_rate=3e-4, clip_grad=0.5,
            gamma=0.99, entropy=0.0,
            normalize_adv=False,
            returns_estimator=None,
            gae_lambda=0.9
    ):
        super().__init__(model, learning_rate, clip_grad)
        self.gamma = gamma
        self.entropy = entropy
        self.normalize_adv = normalize_adv
        self.returns_estimator = ReturnsEstimator(returns_estimator, gamma, gae_lambda)

    def optimize_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norms = {}
        if self.model.obs_encoder is not None:
            for name, child in self.model.obs_encoder.named_children():
                gradient_norms['encoder.' + name + '_grad_norm'] = torch.nn.utils.clip_grad_norm_(
                    child.parameters(), self.clip_grad
                ).item()
        gradient_norms['actor_grad_norm'] = torch.nn.utils.clip_grad_norm_(
            self.model.actor_critic.actor.parameters(), self.clip_grad
        ).item()
        gradient_norms['critic_grad_norm'] = torch.nn.utils.clip_grad_norm_(
            self.model.actor_critic.critic.parameters(), self.clip_grad
        ).item()
        self.optimizer.step()
        return gradient_norms

    def _update_reward_normalizer_scaler(self, data_dict):
        rewards_t = data_dict.get('rewards')
        returns_t = data_dict.get('returns')
        mask = data_dict.get('mask')

        if self.model.reward_normalizer is not None:
            self.model.reward_normalizer.update(rewards_t, mask)
            rewards_t = self.model.reward_normalizer(rewards_t)

        if self.model.reward_scaler is not None:
            self.model.reward_scaler.update(returns_t, mask)
            rewards_t = self.model.reward_scaler(rewards_t)
        data_dict['rewards'] = rewards_t
        return data_dict

    def update_obs_normalizer(self, data_dict):
        if self.model.obs_normalizer is not None:
            # select all observations except the last to correctly sum with mask.
            observation_t = data_dict.get('observations')[:-1]
            mask = data_dict.get('mask')
            self.model.obs_normalizer.update(observation_t, mask)

    def _train_fn(self, rollout):
        raise NotImplementedError

    def train(self, data_dict):
        data_dict = self._update_reward_normalizer_scaler(data_dict)

        train_fn_result, train_fn_time = time_it(self._train_fn)(data_dict)
        if isinstance(train_fn_result, tuple):
            result_log, time_log = train_fn_result
        else:  # i.e. result is one dict
            result_log = train_fn_result
            time_log = dict()
        time_log['train_on_rollout'] = train_fn_time

        self.update_obs_normalizer(data_dict)

        return result_log, time_log
