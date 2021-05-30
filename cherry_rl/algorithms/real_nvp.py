# 'simplified' version of https://github.com/karpathy/pytorch-normalizing-flows/
import math

import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform


from torch_rl.algorithms.nn.actor_critic import MLP
from torch_rl.algorithms.distributions import atanh


class CouplingLayer(nn.Module):
    def __init__(
            self,
            parity,
            input_size, hidden_size, action_size,
            n_layers, activation_str
    ):
        super().__init__()
        self._parity = parity

        self._half_size = action_size // 2

        self._s_t_net = MLP(
            self._half_size + input_size, hidden_size, action_size,
            n_layers, activation_str,
            output_gain=1.0
        )

    def f(self, x, obs):
        x0, x1 = x[..., :self._half_size], x[..., self._half_size:]
        if self._parity:
            x0, x1 = x1, x0
        scale_translate = self._s_t_net(torch.cat([x0, obs], dim=-1))
        scale, translate = torch.split(scale_translate, self._half_size, dim=-1)
        scale = torch.clamp(scale, -5, +5)
        z0 = x0
        z1 = x1 * torch.exp(scale) + translate
        if self._parity:
            z0, z1 = z1, z0
        y = torch.cat([z0, z1], dim=-1)
        log_det = torch.sum(scale, dim=-1)
        return y, log_det

    def g(self, z, obs):
        z0, z1 = z[..., :self._half_size], z[..., self._half_size:]
        if self._parity:
            z0, z1 = z1, z0
        scale_translate = self._s_t_net(torch.cat([z0, obs], dim=-1))
        scale, translate = torch.split(scale_translate, self._half_size, dim=-1)
        scale = torch.clamp(scale, -5, +5)
        x0 = z0
        x1 = (z1 - translate) * torch.exp(-scale)
        if self._parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=-1)
        # probably there must be (-scale), I am not sure.
        log_det = torch.sum(scale, dim=-1)
        return x, log_det


class RealNVPDistribution(nn.Module):
    def __init__(
            self,
            input_size, hidden_size, action_size,
            n_coupling_layers=5,
            coupling_layer_n_layers=2, coupling_layer_activation_str='relu',
            squeeze=False
    ):
        """RealNVP policy distribution.

        Each coupling layer is conditioned on state.
        :param input_size:
        :param hidden_size:
        :param action_size:
        """
        super().__init__()
        self.has_state = True

        self._action_pad = False
        if action_size % 2 != 0:
            action_size += 1
            self._action_pad = True

        self._action_size = action_size
        prior_mean = torch.zeros(action_size, dtype=torch.float32)
        prior_sigma = torch.ones(action_size, dtype=torch.float32)
        self.prior_distribution = Normal(prior_mean, prior_sigma)

        self.layers = nn.ModuleList(
            [
                CouplingLayer(i % 2, input_size, hidden_size, action_size,
                              coupling_layer_n_layers, coupling_layer_activation_str)
                for i in range(n_coupling_layers)
            ]
        )
        self.squeeze = squeeze  # apply tanh to the output

    def _prior_sample(self, *sizes, deterministic):
        if deterministic:
            z = self.prior_distribution.mean.expand((*sizes, -1))
        else:
            z = self.prior_distribution.sample((*sizes,))

        log_prob = self.prior_distribution.log_prob(z).sum(-1)
        return z, log_prob

    def f(self, state_info, action):
        # mapping from action space (a) to latent space (z)
        if self._action_pad:
            zeros = torch.zeros_like(action[..., -1:])
            action = torch.cat([action, zeros], dim=-1)

        z = action
        log_prob = 0

        for layer in self.layers:
            z, layer_log_prob = layer.f(z, state_info)
            log_prob += layer_log_prob

        log_prob += self.prior_distribution.log_prob(z).sum(-1)
        return z, log_prob

    def g(self, state_info, deterministic):
        # mapping from latent space (z) to action space (a)
        z, log_prob = self._prior_sample(*state_info.size()[:-1], deterministic=deterministic)

        x = z
        for layer in reversed(self.layers):
            x, layer_log_prob = layer.g(x, state_info)
            log_prob += layer_log_prob

        if self._action_pad:
            x = x[..., :-1]

        return x, log_prob

    def sample(self, state_info, deterministic):
        action, log_prob = self.g(state_info, deterministic)

        if self.squeeze:
            action = torch.tanh(action)
            action = torch.clamp(action, -0.999, +0.999)
            # tanh does not influence log-prob gradient. Or is it?
        return action, log_prob

    def log_prob(self, state_info, action):
        if self.squeeze:
            action = torch.clamp(action, -0.999, +0.999)
            action = atanh(action)

        _, log_prob = self.f(state_info, action)
        return log_prob

    def mean(self, state_info):
        # mean of the distribution, i.e. sample with prior z = 0.
        return self.sample(state_info, True)

    def entropy(self, state_info):
        # action, log_prob = self.sample(state_info, deterministic=False)
        action, log_prob = self.g(state_info, deterministic=False)
        if self.squeeze:
            # 3.8002 = atanh(0.999)
            action = torch.clamp(action, -3.8002, 3.8002)
            log_d_tanh = math.log(4.0) - 2 * action - 2 * nn.functional.softplus(-2 * action)
            log_prob -= log_d_tanh.sum(-1)
        entropy = -log_prob
        return entropy

    def entropy_uniform(self, state_info):
        u = Uniform(-0.999, +0.999)
        uniform_sample = u.sample((*state_info.size()[:-1], self._action_size))
        log_prob = self.log_prob(state_info, uniform_sample)
        log_prob -= torch.log(1 - uniform_sample.pow(2)).sum(-1)
        entropy = -log_prob
        return entropy

    # def forward(self, state_info, deterministic):
    #     return self.sample(state_info, deterministic)
