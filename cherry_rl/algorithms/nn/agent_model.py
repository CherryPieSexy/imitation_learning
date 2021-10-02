import torch

from cherry_rl.algorithms.normalization import RunningMeanStd


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
            make_policy_distribution,
            obs_normalizer_size=None,
            make_obs_encoder=None,
            reward_normalizer_size=None,
            reward_scaler_size=None,
            value_normalizer_size=None,
    ):
        """
        :param device:
        :param make_actor_critic: actor-critic neural network factory.
               actor-critic nn usage:
                   policy, value, hidden_state = nn(obs)
                   obs.size() = (T, B, dim(obs))
                   policy.size() == (T, B, dim(A) or 2 * dim(A))
                   value.size() == (T, B, dim(value))
        :param make_policy_distribution: policy distribution factory.
        :param obs_normalizer_size: observation normalizer size, None or int.
                                    If None then no normalization is applied to observations.
        :param make_obs_encoder: observation encoder factory.
        :param reward_normalizer_size: reward normalizer normalizer size, None or int.
        :param reward_scaler_size: reward scaler normalizer size, None or int.
        :param value_normalizer_size: value function normalizer normalizer size, None or int.
        """
        super().__init__()
        self.device = device
        self.obs_normalizer = self._make_normalizer(obs_normalizer_size)
        # self.obs_normalizer = RunningMeanStd(size=(24,))
        self.obs_encoder = make_obs_encoder().to(device) if make_obs_encoder is not None else None
        self.reward_normalizer = self._make_normalizer(reward_normalizer_size)
        self.reward_scaler = self._make_normalizer(reward_scaler_size)
        # WARNING: value normalizer should be updated
        # when target value (i.e. returns) is computed.
        self.value_normalizer = self._make_normalizer(value_normalizer_size)

        self.actor_critic = make_actor_critic().to(device)

        if self.reward_scaler is not None:
            self.reward_scaler.subtract_mean = False

        self.recurrent = hasattr(self.obs_encoder, 'recurrent')

        self.pi_distribution = make_policy_distribution()

    def _make_normalizer(self, size):
        return RunningMeanStd(size=(size,)).to(self.device) if size is not None else None

    def preprocess_observation(self, observation, hidden_state):
        result = observation
        if self.obs_normalizer is not None:
            result = self.obs_normalizer.normalize(result)
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

    def act(self, observation, memory, deterministic, with_grad=False):
        observation_t = self.t(observation)
        if with_grad:
            policy, value, memory = self.forward(observation_t, memory)
            # RealNVP requires sampling with 'no_grad'.
            action, log_prob = self.pi_distribution.sample(policy, deterministic)
        else:
            with torch.no_grad():
                policy, value, memory = self.forward(observation_t, memory)
                # RealNVP requires sampling with 'no_grad'.
                action, log_prob = self.pi_distribution.sample(policy, deterministic)

        result = {
            # 'policy': policy, 'value': value,
            'action': action, 'log_prob': log_prob,
            'memory': memory
        }
        return result

    def reset_memory_by_ids(self, memory, ids):
        if self.recurrent:
            return self.obs_encoder.reset_memory_by_ids(memory, ids)
        return memory
