import math
import torch
import torch.nn.functional as fun
import torch.distributions as dist


from algorithms.real_nvp import RealNVP


# Distributions is used to sample actions or states, compute log-probs and entropy
class Distribution:
    def sample(self, parameters, deterministic):
        raise NotImplementedError

    def log_prob(self, parameters, sample):
        raise NotImplementedError

    def entropy(self, *args, **kwargs):
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

    def entropy(self, parameters, *args, **kwargs):
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

    def entropy(self, *args, **kwargs):
        entropy = super().entropy(*args, **kwargs)
        return entropy.sum(-1)


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

    def entropy(self, parameters, *args, **kwargs):
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

    @staticmethod
    def _env_to_agent(action):
        # atanh
        action = torch.clamp(action, -0.9999, +0.9999)
        # noinspection PyTypeChecker
        action = (1.0 + action) / (1.0 - action)
        action = 0.5 * torch.log(action)
        return action

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
        z = self._env_to_agent(sample)
        log_prob = distribution.log_prob(z)
        return log_prob.sum(-1)

    def entropy(self, parameters, sample, **kwargs):
        log_prob = self.log_prob(parameters, sample)
        log_d_tanh = torch.log(1 - sample.pow(2)).sum(-1)
        entropy = -log_prob + log_d_tanh

        return entropy


class Normal(TanhNormal):
    # same as TanhNormal, but without tanh and atanh
    @staticmethod
    def _env_to_agent(action):
        return action

    @staticmethod
    def _agent_to_env(action):
        return action

    def entropy(self, parameters, *args, **kwargs):
        mean, sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, sigma)
        entropy = distribution.entropy()
        return entropy.sum(-1)


distributions_dict = {
    'Categorical': Categorical,
    'GumbelSoftmax': GumbelSoftmax,
    'Bernoulli': Bernoulli,
    'Beta': Beta,
    'WideBeta': WideBeta,
    'TanhNormal': TanhNormal,
    'Normal': Normal,
    'RealNVP': RealNVP
}
