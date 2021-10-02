import torch
import torch.nn as nn
import numpy as np


class RunningMeanStd(nn.Module):
    def __init__(self, clip=10.0, subtract_mean=True, size=(1,)):
        super().__init__()
        self.clip = clip
        self.subtract_mean = subtract_mean
        self.register_buffer('mean', torch.zeros(size, dtype=torch.float32))
        self.register_buffer('var', torch.ones(size, dtype=torch.float32))
        self.register_buffer('count', torch.zeros((1,), dtype=torch.float32))

    def forward(self, x):
        if self.subtract_mean:
            x = (x - self.mean)
        x = x / torch.clamp_min(torch.sqrt(self.var), 1e-6)
        x = torch.clamp(x, -self.clip, self.clip)
        return x

    def normalize(self, x):
        """
        This method is used for observation/reward normalization.

        :param x: torch.Tensor or dict of torch.Tensor-s.
        :return: normalized x.
        """
        if type(x) is dict:
            x_ = torch.cat([v for _, v in x.items()], dim=-1)
            shapes = np.cumsum([0] + [v.shape[-1] for k, v in x.items()], axis=-1)
            slices = {k: (shapes[i], shapes[i + 1]) for i, k in enumerate(x.keys())}
        else:
            x_ = x

        x_ = self.forward(x_)

        if type(x) is dict:
            # noinspection PyUnboundLocalVariable
            x_ = {k: x_[..., slices[k][0]:slices[k][1]] for k in x.keys()}

        return x_

    def denormalize(self, x):
        # this method is only used for value, dict support is not necessary.
        # x = self.mean + x * torch.clamp_min(torch.sqrt(self.var), 1e-6)
        x = x * torch.clamp_min(torch.sqrt(self.var), 1e-6)
        return x

    def update(self, x, mask):
        if self.training:
            if type(x) is dict:
                x = torch.cat([v for _, v in x.items()], dim=-1)
            x_dim = x.size(-1)
            x = x.detach().view(-1, x_dim)
            if mask is not None:
                mask = mask.to(torch.bool).view(-1).unsqueeze(-1)
                x = torch.masked_select(x, mask).view(-1, x_dim)

            batch_mean = x.mean(0)
            batch_var = x.var(0)
            batch_count = x.size(0)

            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            # new_mean = self.mean + delta * batch_count / total_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

            self.mean += delta * batch_count / total_count
            self.var *= 0
            self.var += m_2 / total_count
            # self.var = m_2 / total_count
            self.count += batch_count

    # for some reason default method loads only the first element from tensor.
    def _load_from_state_dict(self, state_dict, prefix, *args):
        self.mean = state_dict[prefix + 'mean']
        self.var = state_dict[prefix + 'var']
        self.count = state_dict[prefix + 'count']
