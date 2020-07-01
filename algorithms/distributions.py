import math
import torch
import torch.nn.functional as fun
import torch.distributions as dist


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
        if deterministic:
            sample = logits.argmax(dim=-1)
        else:
            distribution = self.dist_fn(logits=logits)
            sample = distribution.sample()
        return sample

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
        if deterministic:
            index = logits.argmax(-1, keepdim=True)
            sample = torch.zeros_like(logits)
            sample.scatter_(-1, index, 1.0)
        else:
            sample = fun.gumbel_softmax(logits, hard=True)
        return sample


class Beta(Distribution):
    # we want samples to be in [-1, +1], but supp(Beta) = [0, 1]
    # rescale actions with y = 2 * x - 1, x ~ Beta
    def __init__(self):
        self.dist_fn = dist.Beta

    @staticmethod
    def _agent_to_env(action):
        # [0, 1] -> [-1, +1]
        action = 2.0 * action - 1.0
        action = torch.clamp(action, -0.999, +0.999)
        return action

    @staticmethod
    def _env_to_agent(action):
        # [-1, +1] -> [0, 1]
        action = torch.clamp(action, -0.999, +0.999)
        return 0.5 * (action + 1.0)

    @staticmethod
    def _convert_parameters(parameters):
        parameters = 1.0 + fun.softplus(parameters)
        action_size = parameters.size(-1) // 2
        alpha, beta = parameters.split(action_size, dim=-1)
        return alpha, beta

    def sample(self, parameters, deterministic):
        alpha, beta = self._convert_parameters(parameters)
        if deterministic:
            # good solution: return median
            # but median can be nan (if alpha == beta == 1.0)
            mean = alpha / (alpha + beta)
            median = (alpha - 1.0) / (alpha + beta - 2.0)
            mask = torch.isnan(median)
            z = torch.masked_scatter(median, mask, mean)
        else:
            distribution = self.dist_fn(alpha, beta)
            z = distribution.sample()
        sample = self._agent_to_env(z)
        return sample

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


class TanhNormal(Distribution):
    # WARNING: for some reason, this distribution produces Nans in policy -_-
    def __init__(self):
        self.dist_fn = dist.Normal

    @staticmethod
    def _agent_to_env(action):
        action = torch.tanh(action)
        # action = torch.clamp(action, -0.999, +0.999)
        return action

    @staticmethod
    def _env_to_agent(action):
        # action = torch.clamp(action, -0.999, +0.999)
        # noinspection PyTypeChecker
        action = (1.0 + action) / (1.0 - action)
        action = 0.5 * torch.log(action)
        return action

    @staticmethod
    def _convert_parameters(parameters):
        action_size = parameters.size(-1) // 2
        mean, log_sigma = parameters.split(action_size, -1)
        # some paper suggest to clip log_std to +-15, however I observe Nan-s with this values
        log_sigma_clamp = torch.clamp(log_sigma, -20, +2)
        # forward: torch.clamp(log_sigma)
        # backward: log_sigma
        log_sigma = log_sigma_clamp.detach() - log_sigma.detach() + log_sigma
        sigma = log_sigma.exp()  # works MUCH better with this 0.5
        return mean, sigma

    def sample(self, parameters, deterministic):
        mean, sigma = self._convert_parameters(parameters)
        if deterministic:
            z = mean
        else:
            distribution = self.dist_fn(mean, sigma)
            z = distribution.sample()
        sample = self._agent_to_env(z)
        return sample

    def log_prob(self, parameters, sample):
        mean, sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, sigma)
        z = self._env_to_agent(sample)
        log_prob = distribution.log_prob(z)
        return log_prob.sum(-1)

    def entropy(self, parameters, sample, **kwargs):
        mean, sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, sigma)
        z = self._env_to_agent(sample)
        log_prob = distribution.log_prob(z)

        # noinspection PyTypeChecker
        d_tanh_dx = math.log(4.0) - 2 * torch.log(z.exp() + (-z).exp())
        entropy = -log_prob + d_tanh_dx
        return entropy.sum(-1)


class Normal(TanhNormal):
    # same as TanhNormal, but without tanh and arc-tanh
    # entropy is nut used inside model...
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
    'Beta': Beta,
    'TanhNormal': TanhNormal,
    'Normal': Normal
}
