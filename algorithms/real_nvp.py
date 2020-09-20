# simplified version of https://github.com/karpathy/pytorch-normalizing-flows/
import torch
import torch.nn as nn
from torch.distributions import Normal


from algorithms.nn.actor_critic import init


class MLP(nn.Module):
    # t and s networks
    def __init__(self, input_size, hidden_size):
        super().__init__()
        gain = nn.init.calculate_gain('leaky_relu', 0.2)

        self.net = nn.Sequential(
            init(nn.Linear(input_size, hidden_size), gain=gain), nn.LeakyReLU(0.2),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.LeakyReLU(0.2),
            init(nn.Linear(hidden_size, hidden_size), gain=gain), nn.LeakyReLU(0.2),
            init(nn.Linear(hidden_size, input_size), gain=0.1)
        )

    def forward(self, x):
        return self.net(x)


def _pad(func):
    def wrapper(self, x):
        if self.odd_dimension:
            padding = torch.zeros_like(x[..., -1:])
            x = torch.cat([x, padding], dim=-1)

        y, log_det = func(self, x)

        if self.odd_dimension:
            y = y[..., :-1]
        return y, log_det
    return wrapper


class CouplingLayer(nn.Module):
    def __init__(self, parity, input_size, hidden_size):
        super().__init__()
        self.parity = parity

        self.odd_dimension = False
        if input_size % 2 != 0:
            self.odd_dimension = True
            input_size += 1

        self.half_size = input_size // 2

        self.s_net = MLP(self.half_size, hidden_size)
        self.t_net = MLP(self.half_size, hidden_size)

    @_pad
    def forward(self, x):
        x0, x1 = x[..., :self.half_size], x[..., self.half_size:]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_net(x0)
        t = self.t_net(x0)
        y0 = x0
        y1 = x1 * s.exp() + t
        if self.parity:
            y0, y1 = y1, y0
        y = torch.cat([y0, y1], dim=-1)
        log_det = torch.sum(s, dim=-1)
        return y, log_det

    @_pad
    def backward(self, y):
        # just inverse of Layer, not actual backward
        y0, y1 = y[..., :self.half_size], y[..., self.half_size:]
        if self.parity:
            y0, y1 = y1, y0
        s = self.s_net(y0)
        t = self.t_net(y0)
        x0 = y0
        x1 = (y1 - t) * (-s).exp()
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=-1)
        log_det = torch.sum(-s, dim=-1)
        return x, log_det


class RealNVP(nn.Module):
    def __init__(
            self,
            action_size, hidden_size, n_layers=4,
            std_scale=1.0, train_sigma=False,
            squeeze=False
    ):
        """RealNVP policy distribution

        :param action_size: len of action vector, int
        :param hidden_size: number of units in hidden layers, int
        :param n_layers: number of hidden layers, int
        :param std_scale: initial std value, float, must be positive
        :param train_sigma: if True then std of base (Normal) distribution
                            will be trained, default False
        :param squeeze: if True then Tanh will be applied to the output,
                        default False
        """
        super().__init__()
        self.has_state = True

        prior_mean = torch.zeros(action_size, dtype=torch.float32)
        assert std_scale > 0.0
        prior_sigma = std_scale * torch.ones(action_size, dtype=torch.float32)

        self.prior_mean = prior_mean
        # store sigma in nn.Parameter to save it in checkpoint
        self.prior_log_sigma = nn.Parameter(prior_sigma.log(), requires_grad=train_sigma)
        self.prior_fn = Normal

        self.layers = nn.ModuleList(
            [
                CouplingLayer(i % 2, action_size, hidden_size)
                for i in range(n_layers)
            ]
        )
        self.squeeze = squeeze  # apply tanh to the output

    def _prior_sample(self, time, batch, deterministic):
        distribution = self.prior_fn(
            self.prior_mean, self.prior_log_sigma.exp()
        )

        if deterministic:
            epsilon = self.prior_mean.expand(time, batch, -1)
        else:
            epsilon = distribution.sample((time, batch))

        log_prob = distribution.log_prob(epsilon).sum(-1)
        return epsilon, log_prob

    def _prior_log_prob(self, epsilon):
        distribution = self.prior_fn(
            self.prior_mean, self.prior_log_sigma.exp()
        )
        log_prob = distribution.log_prob(epsilon).sum(-1)
        return log_prob

    def forward(self, state_info, deterministic):
        return self.sample(state_info, deterministic)

    def sample(self, state_info, deterministic):
        time, batch, _ = state_info.size()
        epsilon, log_prob = self._prior_sample(time, batch, deterministic)

        # inject state information __after__ first coupling layer
        layer_input = epsilon
        layer_output, layer_log_prob = self.layers[0](layer_input)
        log_prob += layer_log_prob

        state_info = torch.tanh(state_info)
        layer_input = layer_output + state_info
        for layer in self.layers[1:]:
            layer_output, layer_log_prob = layer(layer_input)
            log_prob += layer_log_prob
            layer_input = layer_output

        if self.squeeze:
            layer_output = torch.tanh(layer_output)
            # tanh does not influence log-prob gradient

        return layer_output, log_prob

    @staticmethod
    def _atanh(x):
        x = torch.clamp(x, -0.9999, +0.9999)
        # noinspection PyTypeChecker
        y = (1.0 + x) / (1.0 - x)
        y = 0.5 * torch.log(y)
        return y

    def log_prob(self, state_info, sample):
        log_prob = torch.zeros_like(state_info[..., 0])

        layer_input = sample
        if self.squeeze:
            layer_input = self._atanh(layer_input)

        for i, layer in enumerate(reversed(self.layers[1:])):
            layer_output, log_det = layer.backward(layer_input)
            log_prob += log_det
            layer_input = layer_output

        state_info = torch.tanh(state_info)
        # noinspection PyUnboundLocalVariable
        layer_input = layer_output - state_info
        # output from the last layer is just epsilon
        layer_output, log_det = self.layers[0].backward(layer_input)
        log_prob += log_det
        prior_log_prob = self._prior_log_prob(layer_output)
        log_prob += prior_log_prob
        return log_prob

    def entropy(self, parameters, sample):
        log_prob = self.log_prob(parameters, sample)
        # sample = torch.clamp(sample, -0.999, +0.999)
        # noinspection PyTypeChecker
        entropy = -log_prob
        if self.squeeze:
            log_d_tanh = torch.log(1 - sample.pow(2)).sum(-1)
            entropy = entropy + log_d_tanh
        return entropy
