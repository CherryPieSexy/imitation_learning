import torch
from torch.nn.utils import clip_grad_norm_

from utils.utils import time_it
from utils.batch_crop import batch_crop
from algorithms.kl_divergence import kl_divergence
from algorithms.distributions import distributions_dict


class AgentInference:
    def __init__(
            self,
            actor_critic,
            device,
            distribution,
            obs_encoder=None,
            obs_normalizer=None,
            reward_normalizer=None,
            reward_scaler=None,
            value_normalizer=None,
    ):
        """
        :param actor_critic: actor-critic neural network: policy, value = nn(obs)
               obs.size() = (T, B, dim(obs))
               policy.size() == (T, B, dim(A) or 2 * dim(A))
               value.size() == (T, B, dim(value))
        :param device:
        :param distribution: distribution type, str.
        :param obs_encoder: observation encoder, nn.Module or None.
        :param obs_normalizer: observation embedding normalizer,
                               RunningMeanStd instance or None.
        :param reward_normalizer: reward normalizer,
                                  RunningMeanStd instance or None.
        :param reward_scaler: reward scaler, RunningMeanStd instance or None.
        :param value_normalizer: value function normalizer,
                                 RunningMeanStd instance or None.
        """
        self.obs_encoder = obs_encoder
        self.obs_normalizer = obs_normalizer
        self.reward_normalizer = reward_normalizer
        self.reward_scaler = reward_scaler
        self.value_normalizer = value_normalizer

        self.actor_critic = actor_critic
        self.device = device

        if self.obs_encoder is not None:
            self.obs_encoder.to(device)
        if self.obs_normalizer is not None:
            self.obs_normalizer.to(device)
        if self.reward_normalizer is not None:
            self.reward_normalizer.to(device)
        if self.reward_scaler is not None:
            self.reward_scaler.to(device)
            self.reward_scaler.subtract_mean = False

        # WARNING: value normalizer should be updated
        # when target value (i.e. returns) is computed.
        if self.value_normalizer is not None:
            self.value_normalizer.to(device)

        self.actor_critic.to(device)
        # WARNING: RealNVP distribution is not supported now.
        # I got better results with non-trainable distributions.
        self.pi_distribution_str = distribution
        self.pi_distribution = distributions_dict[distribution]()

    def state_dict(self):
        state_dict = {
            'actor_critic': self.actor_critic.state_dict(),
            'pi_distribution_str': self.pi_distribution_str
        }
        if self.obs_encoder is not None:
            state_dict.update({'obs_encoder': self.obs_encoder.state_dict()})
        if self.obs_normalizer is not None:
            state_dict.update({'obs_normalizer': self.obs_normalizer.state_dict()})
        if self.reward_normalizer is not None:
            state_dict.update({'reward_normalizer': self.reward_normalizer.state_dict()})
        if self.reward_scaler is not None:
            state_dict.update({'reward_scaler': self.reward_scaler.state_dict()})
        if self.value_normalizer is not None:
            state_dict.update({'value_normalizer': self.value_normalizer.state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        state_pi_distribution = state_dict['pi_distribution_str']
        if self.pi_distribution_str != state_pi_distribution:
            raise ValueError(
                f'different policy distributions in '
                f'agent {self.pi_distribution_str} '
                f'and in checkpoint {state_pi_distribution}'
            )
        if self.obs_encoder is not None:
            self.obs_encoder.load_state_dict(state_dict['obs_encoder'])
        if self.obs_normalizer is not None:
            self.obs_normalizer.load_state_dict(state_dict['obs_normalizer'])
        if self.reward_normalizer is not None:
            self.reward_normalizer.load_state_dict(state_dict['reward_normalizer'])
        if self.reward_scaler is not None:
            self.reward_scaler.load_state_dict(state_dict['reward_scaler'])
        if self.value_normalizer is not None:
            self.value_normalizer.load_state_dict(state_dict['value_normalizer'])
        self.actor_critic.load_state_dict(state_dict['actor_critic'])

    def load(self, filename, **kwargs):
        checkpoint = torch.load(filename, **kwargs)
        self.load_state_dict(checkpoint)

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, filename)

    def train(self):
        if self.obs_encoder is not None:
            self.obs_encoder.train()
        self.actor_critic.train()

    def eval(self):
        if self.obs_encoder is not None:
            self.obs_encoder.eval()
        self.actor_critic.eval()

    def _t(self, x):
        # observation may be dict itself (in goal-augmented or multi-part observation envs)
        if type(x) is torch.Tensor:
            x_t = x.to(torch.float32)
        elif type(x) is dict:
            x_t = {
                key: self._t(value)
                for key, value in x.items()
            }
        else:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x_t

    def _preprocess_observation(self, observation):
        observation_t = self._t(observation)
        if self.obs_encoder is not None:
            observation_t = self.obs_encoder(observation_t)
        if self.obs_normalizer is not None:
            observation_t = self.obs_normalizer(observation_t)
        return observation_t

    def _get_policy_value(self, observation):
        observation_t = self._preprocess_observation(observation)
        actor_critic_result = self.actor_critic(observation_t)
        policy, value = actor_critic_result['policy'], actor_critic_result['value']
        if self.value_normalizer is not None:
            value = self.value_normalizer.denormalize(value)
        return policy, value

    def act(self, observation, deterministic):
        """
        Method to get an action from the agent.

        :param observation: np.array of observation, shape = [B, dim(obs)]
        :param deterministic: default to False, if True then action will be chosen as policy mean
        :return: dict with policy, value, action and log_prob.
                 Each entity is an np.array with shape = [B, dim(entity)]
        """
        with torch.no_grad():
            policy, value = self._get_policy_value(observation)
            # RealNVP requires 'no_grad' here
            action, log_prob = self.pi_distribution.sample(policy, deterministic)

        policy = policy.cpu().numpy()
        value = value.cpu().numpy()
        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy()

        result = {
            'policy': policy, 'value': value,
            'action': action, 'log_prob': log_prob
        }
        return result

    def log_prob(self, observation, action):
        with torch.no_grad():
            policy, value = self._get_policy_value(observation)
            log_prob = self.pi_distribution.log_prob(policy, self._t(action))
        result = {
            'value': value.cpu().numpy(), 'log_prob': log_prob.cpu().numpy()
        }
        return result


# base class for trainable agents
class AgentTrain(AgentInference):
    def __init__(
            self,
            *args,
            learning_rate=3e-4, gamma=0.99, entropy=0.0, clip_grad=0.5,
            normalize_adv=False, returns_estimator='gae',
            gae_lambda=0.9, image_augmentation_alpha=0.0,
            **kwargs
    ):
        """
        :param *args, **kwargs: AgentInference parameters
        :param normalize_adv: True or False
        :param returns_estimator: '1-step', 'n-step', 'gae'
        :param lr, gamma, entropy, clip_grad: learning hyper-parameters
        :param gae_lambda: gae lambda, optional
        :param image_augmentation_alpha: if > 0 then additional
                                         alpha * D_KL(pi, pi_aug) loss term will be used
        """
        super().__init__(*args, **kwargs)
        parameters_to_optimize = self.actor_critic.parameters()
        if self.obs_encoder is not None:
            parameters_to_optimize = list(parameters_to_optimize) + \
                                     list(self.obs_encoder.parameters())
        self.optimizer = torch.optim.Adam(parameters_to_optimize, learning_rate)

        self.gamma = gamma
        self.entropy = entropy
        self.clip_grad = clip_grad

        self.normalize_adv = normalize_adv
        self.returns_estimator = returns_estimator
        self.gae_lambda = gae_lambda
        self.image_augmentation_alpha = image_augmentation_alpha

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({'optimizer': self.optimizer.state_dict()})
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _one_step_returns(self, next_value, rewards, not_done):
        returns = rewards + self.gamma * not_done * next_value
        return returns

    def _n_step_returns(self, next_value, rewards, not_done):
        rollout_len = rewards.size(0)
        last_value = next_value[-1]
        returns = []
        for t in reversed(range(rollout_len)):
            last_value = rewards[t] + self.gamma * not_done[t] * last_value
            returns.append(last_value)
        returns = torch.stack(returns[::-1])
        return returns

    def _gae(self, value, next_value, rewards, not_done):
        rollout_len = rewards.size(0)
        value = torch.cat([value, next_value[-1:]], dim=0)
        gae = 0
        returns = []
        for t in reversed(range(rollout_len)):
            delta = rewards[t] + self.gamma * not_done[t] * value[t + 1] - value[t]
            gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
            returns.append(gae + value[t])
        returns = torch.stack(returns[::-1])
        return returns

    def _estimate_returns(self, value, next_value, rewards, is_done):
        not_done = 1.0 - is_done
        with torch.no_grad():  # returns should not have gradients in any case!
            if self.returns_estimator == '1-step':
                returns = self._one_step_returns(next_value, rewards, not_done)
            elif self.returns_estimator == 'n-step':
                returns = self._n_step_returns(next_value, rewards, not_done)
            elif self.returns_estimator == 'gae':
                returns = self._gae(value, next_value, rewards, not_done)
            else:
                raise ValueError('unknown returns estimator')
        return returns.detach()

    @staticmethod
    def _normalize_advantage(advantage):
        # advantages normalized across batch dimension, I think this is correct
        mean = advantage.mean()
        std = advantage.std()
        advantage = (advantage - mean) / (std + 1e-8)
        return advantage

    def _optimize_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # actor_critic_grad_norm = clip_grad_norm_(
        #     self.actor_critic.parameters(), self.clip_grad
        # )
        actor_grad_norm = clip_grad_norm_(
            self.actor_critic.actor.parameters(), self.clip_grad
        )
        critic_grad_norm = clip_grad_norm_(
            self.actor_critic.critic.parameters(), self.clip_grad
        )
        result = {
            # 'actor_critic_grad_norm': actor_critic_grad_norm,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }
        if self.obs_encoder is not None:
            encoder_grad_norm = clip_grad_norm_(
                self.obs_encoder.parameters(), self.clip_grad
            )
            result.update({'encoder_grad_norm': encoder_grad_norm})
        self.optimizer.step()
        return result

    def _compute_returns_advantage(self, values, rewards, is_done):
        value, next_value = values[:-1], values[1:].detach()

        # reward and not_done must be unsqueezed for correct addition with value
        if len(rewards.size()) == 2:
            rewards = rewards.unsqueeze(-1)
        if len(is_done.size()) == 2:
            is_done = is_done.unsqueeze(-1)

        # returns goes into value loss and so must be kept vectorized to train multi-head critic,
        # but advantage used only for policy update and must be summed along last dim.
        returns = self._estimate_returns(value, next_value, rewards, is_done)
        advantage = (returns - value).sum(-1).detach()
        return returns, advantage

    def _rollout_to_tensor(self, rollout):
        for key, value in rollout.items():
            rollout[key] = self._t(value)
        return rollout

    def _train_fn(self, rollout):
        raise NotImplementedError

    def _image_augmentation_loss(self, observations, policy, value):
        if self.image_augmentation_alpha > 0.0:
            policy, value = policy.detach(), value.detach()

            observations_aug = batch_crop(observations)
            policy_aug, value_aug = self._get_policy_value(observations_aug)
            policy_aug, value_aug = policy_aug[:-1], value_aug[:-1]
            policy_div = kl_divergence(self.pi_distribution_str, policy, policy_aug).mean()
            value_div = 0.5 * ((value - value_aug) ** 2).mean()
            augmentation_loss = self.image_augmentation_alpha * (policy_div + value_div)
            result_dict = {
                'augmentation_policy_div': policy_div.item(),
                'augmentation_value_div': value_div.item()
            }
            return augmentation_loss, result_dict
        else:
            return 0.0, dict()

    def _update_reward_normalizer_scaler(self, rewards, returns):
        # for reward normalization/scaling after normalizer/scaler update is ok
        if self.reward_normalizer is not None:
            rewards_t = self._t(rewards)
            self.reward_normalizer.update(rewards_t)
            rewards = self.reward_normalizer(rewards_t)

        if self.reward_scaler is not None:
            returns_t = self._t(returns)
            self.reward_scaler.update(returns_t)
            rewards = self.reward_scaler(returns_t)
        return rewards

    def _update_obs_normalizer(self, observations):
        # first normalize observations and then update normalizer
        # since I mainly focused on on-policy methods
        if self.obs_normalizer is not None:
            observation_t = self._t(observations)
            if self.obs_encoder is not None:
                observation_t = self.obs_encoder(observation_t)
            self.obs_normalizer.update(observation_t)

    def train_on_rollout(self, rollout, do_train=True):
        rollout['rewards'] = self._update_reward_normalizer_scaler(
            rollout.get('rewards', None), rollout.get('returns', None)
        )

        if do_train:
            train_fn_result, train_fn_time = time_it(self._train_fn)(rollout)
            if isinstance(train_fn_result, tuple):
                result_log, time_log = train_fn_result
            else:  # i.e. result is one dict
                result_log = train_fn_result
                time_log = dict()
            time_log['train_on_rollout'] = train_fn_time
        else:
            result_log, time_log = {}, {}

        self._update_obs_normalizer(rollout['observations'])

        return result_log, time_log
