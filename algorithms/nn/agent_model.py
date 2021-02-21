import torch

from algorithms.normalization import RunningMeanStd
from algorithms.distributions import distributions_dict, TupleDistribution


class AgentModel(torch.nn.Module):
    """
    Full agent model. Includes:
        normalizers: observation, reward, value;
        observation encoder (any nn.Module),
        actor-critic module,
        policy distribution.
    """
    def __init__(
            self,
            device,
            make_actor_critic,
            distribution_str,
            distribution_size=None,
            make_obs_normalizer=False,
            make_obs_encoder=None,
            make_reward_normalizer=False,
            make_reward_scaler=False,
            make_value_normalizer=False,
    ):
        """
        :param device:
        :param make_actor_critic: actor-critic neural network factory.
               actor-critic nn usage:
                   policy, value, hidden_state = nn(obs)
                   obs.size() = (T, B, dim(obs))
                   policy.size() == (T, B, dim(A) or 2 * dim(A))
                   value.size() == (T, B, dim(value))
        :param distribution_str: distribution type, str or tuple of str.
        :param distribution_size: size (number of parameters) for each distribution.
        :param make_obs_normalizer: observation normalizer factory, bool.
                                    normalization is applied to observations.
        :param make_obs_encoder: observation encoder factory.
        :param make_reward_normalizer: reward normalizer factory.
        :param make_reward_scaler: reward scaler factory.
        :param make_value_normalizer: value function normalizer factory.
        """
        super().__init__()
        self.device = device
        self.obs_normalizer = self._make(make_obs_normalizer)
        self.obs_encoder = make_obs_encoder().to(device) if make_obs_encoder is not None else None
        self.reward_normalizer = self._make(make_reward_normalizer)
        self.reward_scaler = self._make(make_reward_scaler)
        # WARNING: value normalizer should be updated
        # when target value (i.e. returns) is computed.
        self.value_normalizer = self._make(make_value_normalizer)

        self.actor_critic = make_actor_critic().to(device)

        if self.reward_scaler is not None:
            self.reward_scaler.subtract_mean = False

        self.recurrent = hasattr(self.obs_encoder, 'recurrent')

        # WARNING: RealNVP distribution is not supported now.
        # I got better results with non-trainable distributions.
        self.pi_distribution_str = distribution_str
        if type(distribution_str) is tuple:
            self.pi_distribution = TupleDistribution(distribution_str, distribution_size)
        else:
            self.pi_distribution = distributions_dict[distribution_str]()

    @staticmethod
    def _make(factory):
        return RunningMeanStd() if factory else None

    def preprocess_observation(self, observation, hidden_state):
        result = observation
        if self.obs_normalizer is not None:
            result = self.obs_normalizer.normalize(observation)
        if self.obs_encoder is not None:
            if self.recurrent:
                result, hidden_state = self.obs_encoder(result, hidden_state)
            else:
                result = self.obs_encoder(result)
        return result, hidden_state

    def forward(self, observation, memory):
        observation_t, memory = self.preprocess_observation(observation, memory)
        actor_critic_result = self.actor_critic(observation_t)
        policy, value = actor_critic_result['policy'], actor_critic_result['value']
        if self.value_normalizer is not None:
            value = self.value_normalizer.denormalize(value)
        return policy, value, memory

    def t(self, x):
        # observation may be dict itself (in goal-augmented or multi-part observation envs)
        if type(x) is torch.Tensor:
            x_t = x.to(torch.float32)
            x_t = x_t.to(self.device)
        elif type(x) is dict:
            x_t = {
                key: self.t(value)
                for key, value in x.items()
            }
        elif type(x) is tuple:
            x_t = tuple([self.t(x[i]) for i in range(len(x))])
        elif x is None:
            x_t = x
        else:
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x_t

    def act(self, observation, memory, deterministic):
        observation_t = self.t(observation)
        with torch.no_grad():
            policy, value, memory = self.forward(observation_t, memory)
        # RealNVP requires sampling with 'no_grad', but it is not supported now anyway.
        action, log_prob = self.pi_distribution.sample(policy, deterministic)

        result = {
            'policy': policy, 'value': value,
            'action': action, 'log_prob': log_prob,
            'memory': memory
        }
        result.update({'memory': memory})
        return result

    def reset_memory_by_ids(self, memory, ids):
        if self.recurrent:
            return self.obs_encoder.reset_memory_by_ids(memory, ids)
        return memory
