import torch
import numpy as np


def _env_num(observation):
    if type(observation) is dict:
        keys = list(observation.keys())
        env_num = observation[keys[0]].shape[0]
    else:
        env_num = observation.shape[0]
    return env_num


class Rollout:
    _keys = [
        'observation', 'reward', 'return', 'done', 'info',  # from env.
        'policy', 'value', 'action', 'log_prob'  # from agent.
    ]
    _plurals = {
        'action': 'actions',
        'reward': 'rewards',
        'return': 'returns',
        'done': 'is_done',
    }
    _singles = {v: k for k, v in _plurals.items()}

    def __init__(self, observation, memory, recurrent):
        self._initial_observation = self._x_to_tensor(observation)
        self._initial_memory = memory
        self._recurrent = recurrent

        self._data = []
        self._data_dict = None
        self._alive_env = [True for _ in range(_env_num(observation))]
        self._mask = []

    @staticmethod
    def _unsqueeze(v):
        if v.dim() == 1:
            return v.unsqueeze(-1)
        return v

    def _x_to_tensor(self, x):
        if type(x) is np.ndarray:
            return torch.tensor(x, dtype=torch.float32)
        elif type(x) is dict:
            return {k: self._x_to_tensor(v) for k, v in x.items()}
        else:
            return x

    def append(self, x):
        x = {k: self._x_to_tensor(v) for k, v in x.items()}
        done, info = x['done'], x['info']
        # info is a tuple of dicts with len = num_envs
        truncated = 1.0 - torch.tensor(
            [i.pop('TimeLimit.truncated', False) for i in info],
            dtype=torch.float32
        )
        self._mask.append(truncated)
        if self._recurrent:
            self._mask[-1] = torch.min(
                self._mask[-1], torch.as_tensor(self._alive_env, dtype=torch.float32)
            )
        self._alive_env = 1.0 - done

        x['reward'] = self._unsqueeze(x['reward'])
        x['return'] = self._unsqueeze(x['return'])
        x['done'] = self._unsqueeze(x['done'])
        self._data.append(x)

    def get(self, key, default=None):
        if key == 'mask':
            return torch.stack(self._mask)
        elif self._data_dict is not None:
            return self._data_dict.get(key, default)
        else:
            result = [d.get(self._singles.get(key, key), default) for d in self._data]
            if result[0] is None:
                return None
            else:
                return torch.stack(result)

    @staticmethod
    def _stack(x):
        if type(x[0]) is tuple:  # only actions may be tuple
            return tuple([
                torch.stack([y[i] for y in x])
                for i in range(len(x[0]))
            ])
        else:
            return torch.stack(x)

    def _cat_observations(self):
        observations = [
            self._initial_observation, *[d['observation'] for d in self._data]
        ]
        if type(observations[0]) is dict:
            return {
                key: torch.stack([obs[key] for obs in observations])
                for key in observations[0].keys()}
        else:
            return torch.stack(observations)

    def as_dict(self):
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
        as_dict['mask'] = torch.stack(self._mask)
        as_dict['memory'] = self._initial_memory
        as_dict['recurrent'] = self._recurrent
        return as_dict
