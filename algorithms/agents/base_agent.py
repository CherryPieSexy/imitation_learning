import torch
from torch.nn.utils import clip_grad_norm_

from utils.utils import time_it
from utils.batch_crop import batch_crop
from algorithms.kl_divergence import kl_divergence
from algorithms.distributions import distributions_dict
from algorithms.normalization import RunningMeanStd


class AgentInference:
    def __init__(
            self,
            nn, device,
            distribution,
            distribution_args
    ):
        self.nn = nn
        self.device = device
        self.nn.to(device)
        self.distribution = distributions_dict[distribution](**distribution_args)
        self.distribution_with_params = False
        if hasattr(self.distribution, 'has_state'):
            self.distribution_with_params = True
            self.distribution.to(device)
        self.obs_normalizer = None

    def load_state_dict(self, state):
        self.nn.load_state_dict(state['nn'])
        if self.distribution_with_params:
            self.distribution.load_state_dict(state['distribution'])

    def load(self, filename, **kwargs):
        checkpoint = torch.load(filename, **kwargs)
        agent_state = checkpoint['agent']
        self.load_state_dict(agent_state)
        if 'obs_normalizer' in checkpoint:
            self.obs_normalizer = RunningMeanStd()
            self.obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])

    def train(self):
        self.nn.train()

    def eval(self):
        self.nn.eval()

    def _t(self, x):
        # observation may be dict itself (in goal-augmented or multi-part observation envs)
        if type(x) is dict:
            x_t = {
                key: torch.tensor(value, dtype=torch.float32, device=self.device)
                for key, value in x.items()
            }
        else:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x_t

    def act(self, observation, deterministic):
        """
        :param observation: np.array of observation, shape = [B, dim(obs)]
        :param deterministic: default to False, if True then action will be chosen as policy mean
        :return: action and log_prob, both np.array with shape = [B, dim(action)]
        """
        if self.obs_normalizer is not None:
            observation = self.obs_normalizer.normalize(observation)
        with torch.no_grad():
            if type(observation) is dict:
                observation = {key: value[None, :] for key, value in observation.items()}
            else:
                observation = [observation]
            nn_result = self.nn(self._t(observation))
            policy, value = nn_result['policy'], nn_result['value']
            # RealNVP requires 'no_grad' here
            action, log_prob = self.distribution.sample(policy, deterministic)

        policy = policy[0].cpu().numpy()
        value = value[0].cpu().numpy()
        action = action[0].cpu().numpy()
        log_prob = log_prob[0].cpu().numpy()

        result = {
            'policy': policy, 'value': value,
            'action': action, 'log_prob': log_prob
        }
        return result

    def log_prob(self, observations, actions):
        with torch.no_grad():
            nn_result = self.nn(self._t(observations))
            policy, value = nn_result['policy'], nn_result['value']
            log_prob = self.distribution.log_prob(policy, self._t(actions))
        result = {
            'value': value, 'log_prob': log_prob
        }
        return result


# base class trainable agents
class AgentTrain:
    def __init__(
            self,
            actor_critic_nn, device,
            distribution, distribution_args,
            learning_rate, gamma, entropy, clip_grad,
            normalize_adv=False, returns_estimator=None,
            gae_lambda=0.95, image_augmentation_alpha=0.0
    ):
        """
        :param actor_critic_nn: actor-critic neural network: policy, value = nn(obs)
               policy.size() == (T, B, dim(A) or 2 * dim(A))
               value.size() == (T, B)  - there is no '1' at the last dimension!
        :param distribution: distribution type, str
        :param distribution_args: distribution arguments, dict. Useful for RealNVP
        :param normalize_adv: True or False
        :param returns_estimator: '1-step', 'n-step', 'gae'
        :param lr, gamma, entropy, clip_grad: learning hyper-parameters
        :param gae_lambda: gae lambda, optional
        :param image_augmentation_alpha: if > 0 then additional
                                         alpha * D_KL(pi, pi_aug) loss term will be used
        """
        self.actor_critic_nn = actor_critic_nn
        self.device = device

        self.policy_distribution_str = distribution
        self.policy_distribution = distributions_dict[distribution](**distribution_args)

        self.actor_critic_nn.to(device)

        parameters_to_optimize = self.actor_critic_nn.parameters()
        self.distribution_with_params = False
        if hasattr(self.policy_distribution, 'has_state'):
            self.distribution_with_params = True
            self.policy_distribution.to(device)
            parameters_to_optimize = list(parameters_to_optimize) + list(self.policy_distribution.parameters())

        self.opt = torch.optim.Adam(parameters_to_optimize, learning_rate)

        self.normalize_adv = normalize_adv
        self.returns_estimator = returns_estimator
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy = entropy
        self.clip_grad = clip_grad
        self.image_augmentation_alpha = image_augmentation_alpha

    def state_dict(self):
        state_dict = {
            'nn': self.actor_critic_nn.state_dict(),
            'opt': self.opt.state_dict()
        }
        if self.distribution_with_params:
            state_dict['distribution'] = self.policy_distribution.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.actor_critic_nn.load_state_dict(state_dict['nn'])
        if self.distribution_with_params:
            self.policy_distribution.load_state_dict(state_dict['distribution'])
        self.opt.load_state_dict(state_dict['opt'])

    def save(self, filename, obs_normalizer=None, reward_normalizer=None):
        """
        Saves nn and optimizer into a file
        :param filename: str
        :param obs_normalizer: RunningMeanStd object
        :param reward_normalizer: RunningMeanStd object
        """
        state_dict = self.state_dict()

        if obs_normalizer is not None:
            state_dict['obs_normalizer'] = obs_normalizer.state_dict()
        if reward_normalizer is not None:
            state_dict['reward_normalizer'] = reward_normalizer.state_dict()
        torch.save(state_dict, filename)

    def load(self, filename, **kwargs):
        checkpoint = torch.load(filename, **kwargs)
        agent_state = checkpoint['agent']
        self.load_state_dict(agent_state)

    def train(self):
        self.actor_critic_nn.train()

    def eval(self):
        self.actor_critic_nn.eval()

    def _t(self, x):
        # observation may be dict itself (in goal-augmented or multipart observation envs)
        if type(x) is dict:
            x_t = {
                key: torch.tensor(value, dtype=torch.float32, device=self.device)
                for key, value in x.items()
            }
        elif type(x) is torch.Tensor:
            x_t = x.clone().detach()
        else:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x_t

    def act(self, observation, deterministic=False):
        """Method to get an action from the agent.

        :param observation: np.array of observation, shape = [B, dim(obs)]
        :param deterministic: default to False, if True then action will be chosen as policy mean
        :return: action and log_prob, both np.array with shape = [B, dim(action)]
        """
        with torch.no_grad():
            if type(observation) is dict:
                observation = {key: [value] for key, value in observation.items()}
            else:
                observation = [observation]
            nn_result = self.actor_critic_nn(self._t(observation))
            policy, value = nn_result['policy'], nn_result['value']
            action, log_prob = self.policy_distribution.sample(policy, deterministic)

        policy = policy[0].cpu().numpy()
        value = value[0].cpu().numpy()
        action = action[0].cpu().numpy()
        log_prob = log_prob[0].cpu().numpy()

        result = {
            'policy': policy, 'value': value,
            'action': action, 'log_prob': log_prob,
        }
        return result

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
        mean = advantage.mean(dim=1, keepdim=True)
        std = advantage.std(dim=1, keepdim=True)
        advantage = (advantage - mean) / (std + 1e-8)
        return advantage

    def optimize_loss(self, loss):
        self.opt.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(
            self.actor_critic_nn.parameters(), self.clip_grad
        )
        self.opt.step()
        return {'actor_critic_grad_norm': gradient_norm}

    def _compute_returns(self, observations, rewards, is_done):
        # TODO: better name for this function.
        #  It not only compute returns, but call nn and compute advantage
        nn_result = self.actor_critic_nn(observations)
        policy, value = nn_result['policy'], nn_result['value']
        policy = policy[:-1]
        value, next_value = value[:-1], value[1:].detach()

        # reward and not_done must be unsqueezed for correct addition with value
        if len(rewards.size()) == 2:
            rewards = rewards.unsqueeze(-1)
        if len(is_done.size()) == 2:
            is_done = is_done.unsqueeze(-1)

        # returns goes into value loss and so must be kept vectorized to train multi-head critic,
        # but advantage goes into policy and must be summed along last dim.
        returns = self._estimate_returns(value, next_value, rewards, is_done)
        advantage = (returns - value).sum(-1).detach()
        if self.normalize_adv:
            advantage = self._normalize_advantage(advantage)
        return policy, value, returns, advantage

    def _rollout_to_tensor(self, rollout):
        for key, value in rollout.items():
            rollout[key] = self._t(value)
        return rollout

    def _train_fn(self, rollout):
        raise NotImplementedError

    def _image_augmentation_loss(self, rollout_t, policy, value):
        if self.image_augmentation_alpha > 0.0:
            observations = rollout_t['observations']
            policy, value = policy.detach(), value.detach()

            observations_aug = batch_crop(observations)
            nn_result = self.actor_critic_nn(observations_aug)
            policy_aug, value_aug = nn_result['policy'], nn_result['value']
            policy_div = kl_divergence(self.policy_distribution_str, policy, policy_aug).mean()
            value_div = 0.5 * ((value - value_aug) ** 2).mean()
            augmentation_loss = self.image_augmentation_alpha * (policy_div + value_div)
            result_dict = {
                'augmentation_policy_div': policy_div.item(),
                'augmentation_value_div': value_div.item()
            }
            return augmentation_loss, result_dict
        else:
            return 0.0, dict()

    def train_on_rollout(self, rollout):
        train_fn_result, train_fn_time = time_it(self._train_fn)(rollout)
        if isinstance(train_fn_result, tuple):
            result_log, time_log = train_fn_result
        else:  # i.e. result is one dict
            result_log = train_fn_result
            time_log = dict()
        time_log['train_on_rollout'] = train_fn_time
        return result_log, time_log
