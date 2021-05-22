import torch
import torch.distributions as dist

from torch_rl.algorithms.distributions import convert_parameters_beta, convert_parameters_normal


def _d_kl_categorical(p, q):
    # categorical distribution parametrized by logits
    logits_diff = torch.log_softmax(p, dim=-1) - torch.log_softmax(q, dim=-1)
    p_probs = torch.softmax(p, dim=-1)
    d_kl = (p_probs * logits_diff).sum(-1)
    return d_kl


def _d_kl_bernoulli(p, q):
    p_probs = torch.clamp(torch.sigmoid(p), 0.001, 0.999)
    q_probs = torch.clamp(torch.sigmoid(q), 0.001, 0.999)
    t1 = p_probs * (p_probs / q_probs).log()
    # noinspection PyTypeChecker,PyUnresolvedReferences
    t2 = (1.0 - p_probs) * ((1.0 - p_probs) / (1.0 - q_probs)).log()
    d_kl = (t1 + t2).mean(-1)
    return d_kl


def _d_kl_normal(p, q):
    # p = target, q = online
    p_mean, p_sigma = convert_parameters_normal(p)
    q_mean, q_sigma = convert_parameters_normal(q)

    mean_diff = ((q_mean - p_mean) / q_sigma).pow(2)
    var_ratio = (p_sigma / q_sigma).pow(2)

    d_kl = 0.5 * (var_ratio + mean_diff - 1 - var_ratio.log())
    return d_kl.mean(-1)


def _d_kl_beta(p, q):
    alpha_p, beta_p = convert_parameters_beta(p)
    alpha_q, beta_q = convert_parameters_beta(q)
    dist_p = dist.Beta(alpha_p, beta_p)
    dist_q = dist.Beta(alpha_q, beta_q)
    d_kl = dist.kl_divergence(dist_p, dist_q).mean(-1)
    return d_kl


# TODO
def _d_kl_tanh_normal(p, q):
    pass


d_kl_dict = {
    'Categorical': _d_kl_categorical,
    'Bernoulli': _d_kl_bernoulli,
    'Normal': _d_kl_normal,
    # 'TanhNormal': _d_kl_normal,  does not work :(
    'Beta': _d_kl_beta
}


def kl_divergence(distribution, p, q):
    return d_kl_dict[distribution](p, q)
