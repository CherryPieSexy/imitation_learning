import numpy as np


class Rollout:
    """
    Simple data storage for on-policy training.

    On-policy algorithm should use it to store data and
    send it to agent for training. Rollout can shuffle data
    for multiple training epochs (i.e. for PPO).
    """
    def __init__(
            self,
            initial_observation, initial_memory,
            recurrent=False
    ):
        if type(initial_observation) is dict:
            keys = initial_observation.keys()
            self.env_num = initial_observation[keys[0]].shape[0]
        else:
            self.env_num = initial_observation.shape[0]

        self._initial_observation = initial_observation
        self._initial_memory = initial_memory
        self._recurrent = recurrent

        self._data = []
        self._data_dict = None
        self._alive_env = [True for _ in range(self.env_num)]
        self._mask = []
        self._keys = [
            'observation', 'reward', 'return', 'done', 'info',  # from env.
            'policy', 'value', 'action', 'log_prob'  # from agent.
        ]
        self._plurals = {
            'action': 'actions',
            'reward': 'rewards',
            'return': 'returns',
            'done': 'is_done',
        }
        self._singles = {v: k for k, v in self._plurals.items()}

    def __len__(self):
        return len(self._data)

    @staticmethod
    def _unsqueeze(x, key):
        value = x.get(key, None)

        if value is not None and len(value.shape) == 1:
            x[key] = value[..., None]
        return x

    def append(self, x):
        done, info = x['done'], x['info']
        # info is a tuple of dicts with len = num_envs
        truncated = 1.0 - np.array(
            [i.pop('TimeLimit.truncated', False) for i in info],
            dtype=np.float32
        )
        self._mask.append(truncated)
        if self._recurrent:
            self._mask[-1] = np.minimum(self._mask[-1], self._alive_env)
        self._alive_env = 1.0 - done

        x = self._unsqueeze(x, 'reward')
        x = self._unsqueeze(x, 'return')
        x = self._unsqueeze(x, 'done')
        self._data.append(x)

    def _cat_observations(self):
        observations = [
            self._initial_observation, *[d['observation'] for d in self._data]
        ]
        if type(observations[0]) is dict:
            return {
                key: np.stack([obs[key] for obs in observations])
                for key in observations[0].keys()}
        else:
            return np.stack(observations)

    @staticmethod
    def _stack(x):
        if type(x[0]) is tuple:
            return tuple([
                np.stack([y[i] for y in x])
                for i in range(len(x[0]))
            ])
        else:
            return np.stack(x)

    def to_dict(self):
        if self._data_dict is not None:
            raise ValueError('Rollout is already converted to dict.')
        keys = [
            k for k in self._data[0].keys()
            if k not in ['observation', 'info']
        ]
        as_dict = {
            self._plurals.get(key, key): self._stack([data[key] for data in self._data])
            for key in keys
        }
        as_dict['observations'] = self._cat_observations()
        self._data_dict = as_dict
        self._mask = np.stack(self._mask)

    def to_tensor(self, to_tensor_fn, device):
        if self._data_dict is None:
            self.to_dict()
        for k, v in self._data_dict.items():
            self._data_dict[k] = to_tensor_fn(v, device)
        self._mask = to_tensor_fn(self._mask, device)
        return self._data_dict

    @property
    def recurrent(self):
        return self._recurrent

    @property
    def as_dict(self):
        return self._data_dict

    @property
    def mask(self):
        if type(self._mask) is list:
            self._mask = np.stack(self._mask)
        return self._mask

    @property
    def memory(self):
        return self._initial_memory

    def get(self, key, default=None):
        if self._data_dict is not None:
            return self._data_dict.get(key, default)
        else:
            result = [d.get(self._singles.get(key, key), default) for d in self._data]
            if result[0] is None:
                return None
            else:
                return np.stack(result)

    def set(self, key, value):
        if self._data_dict is not None:
            self._data_dict[key] = value
        else:
            raise ValueError('Rollout must be converted to dict before calling \'set\' method.')

    @staticmethod
    def _select(select_from, row, col):
        if type(select_from) is dict:
            # observation may be of type dict
            return {key: value[row, col] for key, value in select_from.items()}
        elif type(select_from) is tuple:
            return tuple(value[row, col] for value in select_from)
        else:
            return select_from[row, col]

    def _select_by_row_col(self, rollout, row, col):
        return {k: self._select(v, row, col) for k, v in rollout.items()}

    def _feed_forward_data_generator(self, num_batches):
        time, batch = len(self._data), self.env_num
        num_transitions = time * batch
        batch_size = num_transitions // num_batches

        flatten_indices = np.arange(num_transitions)
        np.random.shuffle(flatten_indices)

        for _, start_id in enumerate(range(0, num_transitions, batch_size)):
            selected_indices = flatten_indices[start_id:start_id + batch_size]
            row = selected_indices // batch
            col = selected_indices - batch * row

            yield \
                self._select_by_row_col(self._data_dict, row, col),\
                self._select(self._mask, row, col),\
                None  # initial memory, shouldn't be used here

    @staticmethod
    def _select_col(select_from, col):
        if type(select_from) is dict:
            return {key: value[:, col] for key, value in select_from.items()}
        elif type(select_from) is tuple:
            # memory should have batch as 2-nd dimension (i.e. index '1').
            return tuple([value[:, col] for value in select_from])
        elif select_from is None:
            return None
        else:
            return select_from[:, col]

    def _select_by_col(self, rollout, col):
        return {k: self._select_col(v, col) for k, v in rollout.items()}

    def _recurrent_data_generator(self, num_sequences):
        batch = self.env_num
        step_size = batch // num_sequences
        batch_indices = np.arange(batch)
        np.random.shuffle(batch_indices)

        for _, time_id in enumerate(range(0, batch, step_size)):
            col = batch_indices[time_id:time_id + step_size]
            data_dict = self._select_by_col(self._data_dict, col)
            data_dict['observations'] = data_dict['observations'][:-1]
            yield \
                data_dict,\
                self._select_col(self._mask, col),\
                self._select_col(self._initial_memory, col)

    def get_data_generator(self, num_batches):
        """
        Creates data generator for multiple optimization steps on one rollout.
        Data generator returns tuples of (rollout_dict, mask, memory).

        :param num_batches: number of elements in one data generator.
        :return: data generator.
        """
        assert self._data_dict is not None, \
            'Rollout should be converted to dict (by to_dict() or to_tensor() method)' \
            'before generating data.'
        if self._recurrent:
            return self._recurrent_data_generator(num_batches)
        else:
            return self._feed_forward_data_generator(num_batches)
