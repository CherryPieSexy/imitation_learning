# simplified version of https://github.com/karpathy/pytorch-normalizing-flows/
import torch
import torch.nn as nn
from torch.distributions import Normal


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

        self._s_t_net = nn.Sequential(
            MLP(
                self._half_size + input_size, hidden_size, action_size,
                n_layers, activation_str,
                output_gain=5.0 / 3.0),
            nn.Tanh()
        )

    def f(self, x, obs):
        x0, x1 = x[..., :self._half_size], x[..., self._half_size:]
        if self._parity:
            x0, x1 = x1, x0
        scale_translate = self._s_t_net(torch.cat([x0, obs], dim=-1))
        scale, translate = torch.split(scale_translate, self._half_size, dim=-1)
        z0 = x0
        z1 = x1 * scale.exp() + translate
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
        x0 = z0
        x1 = (z1 - translate) * (-scale).exp()
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
            n_coupling_layers=4,
            coupling_layer_n_layers=3, coupling_layer_activation_str='relu',
            squeeze=False
    ):
        """RealNVP policy distribution

        :param action_size: len of action vector, int
        :param hidden_size: number of units in hidden layers, int
        :param n_coupling_layers: number of coupling layers, int
        :param squeeze: if True then Tanh will be applied to the output.
        """
        super().__init__()
        self.has_state = True

        self._action_pad = False
        if action_size % 2 != 0:
            action_size += 1
            self._action_pad = True

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
        # mapping from action space to latent space
        if self._action_pad:
            zeros = torch.zeros_like(action[..., -1:])
            action = torch.cat([action, zeros], dim=-1)

        z = action
        log_prob = 0

        if self.squeeze:
            z = atanh(z)

        for layer in self.layers:
            z, layer_log_prob = layer.f(z, state_info)
            log_prob += layer_log_prob

        log_prob += self.prior_distribution.log_prob(z).sum(-1)
        return z, log_prob

    def g(self, state_info, deterministic):
        # mapping from latent space to action space
        z, log_prob = self._prior_sample(*state_info.size()[:-1], deterministic=deterministic)

        x = z
        for layer in reversed(self.layers):
            x, layer_log_prob = layer.g(x, state_info)
            log_prob += layer_log_prob

        if self.squeeze:
            x = torch.tanh(x)
            # tanh does not influence log-prob gradient. Or is it?

        if self._action_pad:
            x = x[..., :-1]

        return x, log_prob

    def sample(self, state_info, deterministic):
        action, log_prob = self.g(state_info, deterministic)
        # print(state_info.mean(), state_info.min(), state_info.max())
        # print('a:', action.mean(), action.min(), action.max(), action.requires_grad)
        # print('l:', log_prob.mean(), log_prob.min(), log_prob.max(), log_prob.requires_grad)
        return action, log_prob

    def log_prob(self, state_info, action):
        _, log_prob = self.f(state_info, action)
        return log_prob

    def mean(self, state_info):
        # mean of the distribution, i.e. sample with prior z = 0.
        return self.sample(state_info, True)

    def entropy(self, state_info):
        action, log_prob = self.sample(state_info, deterministic=False)
        action = torch.clamp(action, -0.999, +0.999)
        # noinspection PyTypeChecker
        entropy = -log_prob
        if self.squeeze:
            log_d_tanh = torch.log(1 - action.pow(2)).sum(-1)
            entropy = entropy + log_d_tanh
        return entropy

    # def forward(self, state_info, deterministic):
    #     return self.sample(state_info, deterministic)
