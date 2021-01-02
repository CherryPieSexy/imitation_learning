import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    def __init__(self, clip=10.0, subtract_mean=True):
        super().__init__()
        self.clip = clip
        self.subtract_mean = subtract_mean
        self.register_buffer('mean', torch.zeros((), dtype=torch.float32))
        self.register_buffer('var', torch.ones((), dtype=torch.float32))
        self.register_buffer('count', torch.zeros((1,), dtype=torch.float32))

    def forward(self, x):
        if self.subtract_mean:
            x = (x - self.mean)
        x = x / torch.clamp_min(torch.sqrt(self.var), 1e-6)
        x = torch.clamp(x, -self.clip, self.clip)
        return x

    def normalize(self, x):
        return self.forward(x)

    def denormalize(self, x):
        x = self.mean + x * torch.clamp_min(torch.sqrt(self.var), 1e-6)
        return x

    def update(self, x):
        x_dim = x.size(-1)
        x = x.detach().view(-1, x_dim)

        batch_mean = x.mean(0)
        batch_var = x.var(0)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['clip'] = self.clip
        state_dict['subtract_mean'] = self.subtract_mean
        return state_dict

    # for some reason default method loads only first element in tensor.
    def load_state_dict(self, state_dict, strict=True):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']
        self.clip = state_dict['clip']
        self.subtract_mean = state_dict['subtract_mean']
