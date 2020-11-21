# code from https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
import numpy as np


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, clip=10.0, epsilon=1e-4, shape=()):
        self.clip = clip
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if type(x) is dict:
            x = np.concatenate([v for k, v in x.items()], axis=1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def update_vec(self, x):
        # it works just fine for scalar reward
        if type(x) is dict:
            time, batch = x[list(x)[0]].shape[:2]  # list(x) returns list of dict keys
            self.update({k: v.reshape((time * batch, -1)) for k, v in x.items()})
        else:
            time, batch = x.shape[:2]
            self.update(x.reshape((time * batch, -1)))

    def normalize(self, x):
        if type(x) is dict:
            x_ = np.concatenate([v for k, v in x.items()], axis=1)
            shapes = np.cumsum([0] + [v.shape[1] for k, v in x.items()])
            slices = {k: (shapes[i], shapes[i + 1]) for i, k in enumerate(x.keys())}
        else:
            x_ = x

        x_ = (x_ - self.mean) / np.maximum(np.sqrt(self.var), 1e-6)
        x_ = np.clip(x_, -self.clip, self.clip)

        if type(x) is dict:
            # noinspection PyUnboundLocalVariable
            x_ = {k: x_[:, slices[k][0]:slices[k][1]] for k in x.keys()}
        return x_

    def normalize_vec(self, x):
        if type(x) is dict:
            time, batch = x[list(x)[0]].shape[:2]
            x_ = self.normalize({k: v.reshape((time * batch, -1)) for k, v in x.items()})
            x_ = {k: v.reshape((time, batch, -1)) for k, v in x_.items()}
        else:
            time, batch = x.shape[:2]
            x_ = self.normalize(x.reshape((time * batch, -1)))
            x_ = x_.reshape((time, batch, -1))
        return x_

    def scale(self, x):
        x = x / np.maximum(np.sqrt(self.var), 1e-6)
        x = np.clip(x, self.clip, self.clip)
        return x

    def scale_vec(self, x):
        time, batch = x.shape[:2]
        x_ = self.scale(x.reshape((time * batch, -1)))
        x_ = x_.reshape((time, batch, -1))
        return x_

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def state_dict(self):
        state_dict = {
            'clip': self.clip,
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.clip = state_dict.get('clip', 10.0)
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']
