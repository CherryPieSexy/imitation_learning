import math
import torch
import torch.nn.functional as fun
import torch.distributions as dist


def atanh(x):
    x = torch.clamp(x, -0.9999, +0.9999)
    # noinspection PyTypeChecker
    y = (1.0 + x) / (1.0 - x)
    y = 0.5 * torch.log(y)
    return y


# Distributions is used to sample actions or states, compute log-probs and entropy
class Distribution:
    def sample(self, parameters, deterministic):
        raise NotImplementedError

    def log_prob(self, parameters, sample):
        raise NotImplementedError

    def entropy(self, parameters):
        raise NotImplementedError


class Categorical(Distribution):
    def __init__(self):
        self.dist_fn = dist.Categorical

    def sample(self, parameters, deterministic):
        logits = parameters
        distribution = self.dist_fn(logits=logits)
        if deterministic:
            sample = logits.argmax(dim=-1)
        else:
            sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        return sample, log_prob

    def log_prob(self, parameters, sample):
        logits = parameters
        distribution = self.dist_fn(logits=logits)
        log_prob = distribution.log_prob(sample)
        return log_prob

    def entropy(self, parameters):
        logits = parameters
        distribution = self.dist_fn(logits=logits)
        entropy = distribution.entropy()
        return entropy


class GumbelSoftmax(Categorical):
    # same as categorical, but actions are one-hot and differentiable
    def __init__(self):
        super().__init__()
        self.dist_fn = dist.OneHotCategorical

    def sample(self, parameters, deterministic):
        logits = parameters
        distribution = self.dist_fn(logits=logits)
        if deterministic:
            index = logits.argmax(-1, keepdim=True)
            sample = torch.zeros_like(logits)
            sample.scatter_(-1, index, 1.0)
        else:
            sample = fun.gumbel_softmax(logits, hard=True)
        log_prob = distribution.log_prob(sample)
        return sample, log_prob


class Bernoulli(Categorical):
    def __init__(self):
        super().__init__()
        self.dist_fn = dist.Bernoulli

    def sample(self, parameters, deterministic):
        logits = parameters
        distribution = self.dist_fn(logits=logits)
        if deterministic:
            sample = (logits > 0).to(torch.float32)
        else:
            sample = distribution.sample()
        log_prob = distribution.log_prob(sample).sum(-1)
        return sample, log_prob

    def log_prob(self, parameters, sample):
        log_prob = super().log_prob(parameters, sample)
        return log_prob.sum(-1)

    def entropy(self, parameters):
        entropy = super().entropy(parameters)
        return entropy.sum(-1)


# TODO: Gumbel Bernoulli


def convert_parameters_beta(parameters):
    parameters = 1.0 + fun.softplus(parameters)
    action_size = parameters.size(-1) // 2
    alpha, beta = parameters.split(action_size, dim=-1)
    return alpha, beta


class Beta(Distribution):
    # we want samples to be in [-1, +1], but supp(Beta) = [0, 1]
    # rescale actions with y = 2 * x - 1, x ~ Beta
    def __init__(self):
        self.dist_fn = dist.Beta
        self._convert_parameters = convert_parameters_beta

    @staticmethod
    def _agent_to_env(action):
        # [0, 1] -> [-1.0, +1.0]
        action = 2.0 * action - 1.0
        action = torch.clamp(action, -0.9999, +0.9999)
        return action

    @staticmethod
    def _env_to_agent(action):
        # [-1.0, +1.0] -> [0, 1]
        action = torch.clamp(action, -0.9999, +0.9999)
        action = (action + 1.0) / 2.0
        return action

    def mean(self, parameters):
        alpha, beta = self._convert_parameters(parameters)
        mean = alpha / (alpha + beta)
        return mean

    def sample(self, parameters, deterministic):
        alpha, beta = self._convert_parameters(parameters)
        distribution = self.dist_fn(alpha, beta)
        if deterministic:
            # mean works better than median in practice
            z = alpha / (alpha + beta)
        else:
            z = distribution.sample()
        log_prob = distribution.log_prob(z)
        sample = self._agent_to_env(z)
        return sample, log_prob.sum(-1)

    def log_prob(self, parameters, sample):
        # do not use rescaling here
        alpha, beta = self._convert_parameters(parameters)
        distribution = self.dist_fn(alpha, beta)
        z = self._env_to_agent(sample)
        log_prob = distribution.log_prob(z)
        return log_prob.sum(-1)

    def entropy(self, parameters):
        # entropy changes because of rescaling:
        # H(y) = H(x) + log(2.0)
        alpha, beta = self._convert_parameters(parameters)
        distribution = self.dist_fn(alpha, beta)
        entropy = distribution.entropy() + math.log(2.0)
        return entropy.sum(-1)


class WideBeta(Beta):
    @staticmethod
    def _agent_to_env(action):
        # [0, 1] -> [-1.2, +1.2]
        action = 2.4 * action - 1.2
        action = torch.clamp(action, -1.1999, +1.1999)
        return action

    @staticmethod
    def _env_to_agent(action):
        # [-1.2, +1.2] -> [0, 1]
        action = torch.clamp(action, -1.1999, +1.1999)
        action = (action + 1.2) / 2.4
        return action


def convert_parameters_normal(parameters):
    half = parameters.size(-1) // 2
    mean, log_sigma = parameters.split(half, -1)
    log_sigma_clamp = torch.clamp(log_sigma, -20, +2)
    sigma = log_sigma_clamp.exp()
    return mean, sigma


class TanhNormal(Distribution):
    def __init__(self):
        self.dist_fn = dist.Normal
        self._convert_parameters = convert_parameters_normal

    @staticmethod
    def _agent_to_env(action):
        action = torch.tanh(action)
        action = torch.clamp(action, -0.9999, +0.9999)
        return action

    def mean(self, parameters):
        mean, _ = self._convert_parameters(parameters)
        mean = torch.tanh(mean)
        return mean

    def sample(self, parameters, deterministic):
        mean, sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, sigma)
        if deterministic:
            z = mean
        else:
            z = distribution.sample()
        log_prob = distribution.log_prob(z)
        sample = self._agent_to_env(z)
        return sample, log_prob.sum(-1)

    def log_prob(self, parameters, sample):
        mean, sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, sigma)
        z = atanh(sample)
        log_prob = distribution.log_prob(z)
        return log_prob.sum(-1)

    def entropy(self, parameters):
        sample, log_prob = self.sample(parameters, deterministic=False)
        log_d_tanh = torch.log(1 - sample.pow(2)).sum(-1)
        entropy = -log_prob + log_d_tanh

        return entropy


class Normal(TanhNormal):
    # same as TanhNormal, but without tanh and atanh
    @staticmethod
    def _env_to_agent(action):
        return action

    def mean(self, parameters):
        mean, _ = self._convert_parameters(parameters)
        return mean

    @staticmethod
    def _agent_to_env(action):
        return action

    def entropy(self, parameters):
        mean, sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, sigma)
        entropy = distribution.entropy()
        return entropy.sum(-1)


class TupleDistribution(Distribution):
    """
    Special multi-part distribution represented as tuple of distributions.
    """
    def __init__(self, distributions, sizes):
        """
        :param distributions: list of strings - names of distributions.
        :param sizes: size (number of parameters) for each distribution.
        """
        self._distributions = [distributions_dict[d]() for d in distributions]
        self._sizes = sizes

    def sample(self, parameters, deterministic):
        parameter_idx = 0
        sample, log_prob = [], 0
        for distribution, size in zip(self._distributions, self._sizes):
            d_sample, d_log_prob = distribution.sample(
                parameters[..., parameter_idx:parameter_idx + size],
                deterministic
            )
            sample.append(d_sample)
            log_prob += d_log_prob
            parameter_idx += size
        return tuple(sample), log_prob

    def log_prob(self, parameters, sample):
        parameter_idx = 0
        log_prob = 0
        for distribution, size, d_sample in zip(self._distributions, self._sizes, sample):
            log_prob += distribution.log_prob(
                parameters[..., parameter_idx:parameter_idx + size],
                d_sample
            )
            parameter_idx += size
        return log_prob

    def entropy(self, parameters):
        parameter_idx = 0
        entropy = 0
        for distribution, size in zip(self._distributions, self._sizes):
            entropy += distribution.entropy(
                parameters[..., parameter_idx:parameter_idx + size]
            )
            parameter_idx += size
        return entropy


distributions_dict = {
    'Categorical': Categorical,
    'GumbelSoftmax': GumbelSoftmax,
    'Bernoulli': Bernoulli,
    'Beta': Beta,
    'WideBeta': WideBeta,
    'TanhNormal': TanhNormal,
    'Normal': Normal,
}
