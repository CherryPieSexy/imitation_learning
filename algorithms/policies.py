import math
import torch
import torch.nn.functional as fun
import torch.distributions as dist


# Policy is used to sample actions, compute log-probs and entropy
# can be separated from agent.
# TODO: Normal, NormalFixedSigma, GumbelCategorical (state-trough with reparametrization
# TODO: it is unclear what should I do with entropy of tanh(Normal).
#  Practical solution - force action to be close to 0, so set entropy to -action.pow(2).mean()


class Categorical:
    def __init__(self):
        self.dist_fn = dist.Categorical

    def sample(self, parameters, deterministic):
        logits = parameters
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            distribution = self.dist_fn(logits=logits)
            action = distribution.sample()
        return action

    def log_prob(self, parameters, action):
        logits = parameters
        distribution = self.dist_fn(logits=logits)
        log_prob = distribution.log_prob(action)
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
        if deterministic:
            index = logits.argmax(-1, keepdim=True)
            action = torch.zeros_like(logits)
            action.scatter_(-1, index, 1.0)
        else:
            action = fun.gumbel_softmax(logits, hard=True)
        return action


class Beta:
    # we want samples to be in [-1, +1], but supp(Beta) = [0, 1]
    # rescale actions with y = 2 * x - 1, x ~ Beta
    def __init__(self):
        self.dist_fn = dist.Beta

    @staticmethod
    def _agent_to_env(action):
        return 2.0 * action - 1.0

    @staticmethod
    def _env_to_agent(action):
        return 0.5 * (action + 1.0)

    @staticmethod
    def _convert_parameters(parameters):
        parameters = 1.0 + torch.log(1.0 + parameters.exp())
        action_size = parameters.size(-1) // 2
        alpha, beta = parameters.split(action_size, dim=-1)
        return alpha, beta

    def sample(self, parameters, deterministic):
        alpha, beta = self._convert_parameters(parameters)
        if deterministic:
            # z = alpha / (alpha + beta)
            # more practical solution - return mode instead of mean
            z = (alpha - 1.0) / (alpha + beta - 2.0)
        else:
            distribution = self.dist_fn(alpha, beta)
            z = distribution.sample()
        action = self._agent_to_env(z)
        return action

    def log_prob(self, parameters, action):
        # log prob changes because of rescaling:
        # log_pi(a) - log(2.0).
        # This rescaling procedure do not influence gradients, so it may be omitted
        alpha, beta = self._convert_parameters(parameters)
        distribution = self.dist_fn(alpha, beta)
        z = self._env_to_agent(action)
        log_prob = distribution.log_prob(z) - math.log(2.0)
        return log_prob.sum(-1)

    def entropy(self, parameters):
        # entropy changes because of rescaling:
        # H(y) = H(x) + log(2.0)
        alpha, beta = self._convert_parameters(parameters)
        distribution = self.dist_fn(alpha, beta)
        entropy = distribution.entropy() + math.log(2.0)
        return entropy.sum(-1)


# For simple task (Pendulum) it is better to use N(mu, 0.1)
class NormalFixedSigma:
    # WARNING: this distribution now is w/o tanh! Is it correct?
    def __init__(self):
        self.dist_fn = dist.Normal
        self.sigma = 0.5

    def sample(self, parameters, deterministic):
        mean = parameters
        if deterministic:
            action = mean
        else:
            sigma = torch.full_like(mean, self.sigma)
            distribution = self.dist_fn(mean, sigma)
            action = distribution.sample()

        return action

    def log_prob(self, parameters, action):
        mean = parameters
        sigma = torch.full_like(mean, self.sigma)
        distribution = self.dist_fn(mean, sigma)
        log_prob = distribution.log_prob(action)
        return log_prob.sum(-1)

    @staticmethod
    def entropy(parameters):
        # increase entropy == make action closer to 0
        mean = parameters
        entropy = -(mean ** 2).sum(-1)
        return entropy


class TanhNormal:
    MIN_LOG_SIGMA = -20.0
    MAX_LOG_SIGMA = 2.0

    def __init__(self):
        self.dist_fn = dist.Normal

    def _convert_parameters(self, parameters):
        mean, log_sigma = parameters
        log_sigma = torch.clamp(log_sigma, self.MIN_LOG_SIGMA, self.MAX_LOG_SIGMA)
        return mean, log_sigma

    def sample(self, parameters, deterministic):
        mean, log_sigma = parameters
        if deterministic:
            z = mean
        else:
            distribution = self.dist_fn(mean, log_sigma.exp())
            z = distribution.sample()
        action = torch.tanh(z)
        return action

    @staticmethod
    def _atanh(x):
        x = torch.clamp(x, -0.999, +0.999)
        # noinspection PyTypeChecker
        z = 0.5 * torch.log((1.0 + x) / (1.0 - x))
        return z

    def log_prob(self, parameters, action):
        # log prob changes because of tanh
        # log_pi(a) - log(d(tanh(z)) / dz)
        mean, log_sigma = self._convert_parameters(parameters)
        distribution = self.dist_fn(mean, log_sigma.exp())
        z = self._atanh(action)
        log_prob_z = distribution.log_prob(z)
        log_prob = log_prob_z - math.log(4.0) + 2.0 * torch.log(z.exp() + (-z).exp())
        return log_prob.sum(-1)

    def entropy(self, parameters):
        # increase entropy = decrease mean + increase variance
        mean, log_sigma = self._convert_parameters(parameters)
        entropy = -(mean ** 2).sum(-1) + 0.01 * log_sigma
        return entropy.sum(-1)


distributions_dict = {
    'Categorical': Categorical,
    'GumbelSoftmax': GumbelSoftmax,
    'Beta': Beta,
    'NormalFixed': NormalFixedSigma,
    'TanhNormal': TanhNormal,
}
