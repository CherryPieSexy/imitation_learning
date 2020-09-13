import torch
import numpy as np

from algorithms.distributions import convert_parameters_normal
from algorithms.agents.policy_gradient import PolicyGradient


class VMPO(PolicyGradient):
    def __init__(
            self,
            *args,
            eps_eta_range,
            eps_mu_range,
            eps_sigma_range,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eta = self._init_param(1.0, True)
        self.eps_eta = self._log_uniform(eps_eta_range)

        param_list = [self.eta]
        if self.policy_distribution_str == 'Categorical':
            self.alpha = self._init_param(5.0, True)
            self.eps_alpha = self._log_uniform(eps_mu_range)
            param_list.append(self.alpha)
        elif self.policy_distribution_str in ['Normal', 'TanhNormal']:
            self.alpha_mu = self._init_param(1.0, True)
            self.alpha_sigma = self._init_param(1.0, True)
            self.eps_mu = self._log_uniform(eps_mu_range)
            self.eps_sigma = self._log_uniform(eps_sigma_range)
            param_list.append(self.alpha_mu)
            param_list.append(self.alpha_sigma)
        else:
            msg = f'V-MPO support only Categorical and Normal distributions, not {self.policy_distribution_str}'
            raise ValueError(msg)
        self.param_opt = torch.optim.Adam(param_list, lr=1e-4)
        self.param_min = self._init_param(1e-8, False)

    @staticmethod
    def _log_uniform(low_high):
        low, high = low_high
        if low < high:
            z = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            z = high
        return z

    def _init_param(self, value, requires_grad):
        param = torch.tensor(
            value,
            dtype=torch.float32,
            device=self.device,
            requires_grad=requires_grad
        )
        return param

    @staticmethod
    def _kl_divergence_categorical(pi_target, pi_online):
        # pi_target and pi is nn outputs before softmax
        p_prob = torch.softmax(pi_target, dim=-1)
        p_log_prob = torch.log_softmax(pi_target, dim=-1)
        q_log_prob = torch.log_softmax(pi_online, dim=-1)
        d_kl = (p_prob * (p_log_prob - q_log_prob)).sum(-1)
        return d_kl

    @staticmethod
    def _kl_divergence_normal(pi_target, pi_online):
        action_size = pi_target.size(-1) // 2
        mean_target, sigma_target = convert_parameters_normal(pi_target)
        mean_online, sigma_online = convert_parameters_normal(pi_online)

        mean_diff = (mean_online - mean_target)
        d_kl_mean = mean_diff ** 2 / sigma_online
        d_kl_mean = 0.5 * d_kl_mean.sum(-1)

        trace = (sigma_target / sigma_online).sum(-1)
        log_det = torch.log(sigma_online).sum(-1) - torch.log(sigma_target).sum(-1)
        d_kl_sigma = 0.5 * (trace - action_size + log_det)
        return d_kl_mean, d_kl_sigma

    @staticmethod
    def _select_top_adv(advantage):
        time, batch = advantage.size()
        half = time * batch // 2
        flat_adv = advantage.flatten()
        top_adv, flat_ids = torch.topk(flat_adv, half)
        # noinspection PyUnresolvedReferences
        row = flat_ids // batch
        col = flat_ids - batch * row
        return row, col, half

    def _policy_loss(self, policy, actions, softmax_adv):
        log_pi_for_actions = self.policy_distribution.log_prob(policy, actions)
        policy_loss = -log_pi_for_actions * softmax_adv.detach()
        return policy_loss.sum()

    @staticmethod
    def _alpha_loss_formula(alpha, eps_alpha, d_kl):
        return alpha * (eps_alpha - d_kl.detach()) + alpha.detach() * d_kl

    def _alpha_loss(self, policy_old, policy):
        if self.policy_distribution_str == 'Categorical':
            d_kl = self._kl_divergence_categorical(policy_old, policy)
            alpha_loss = self._alpha_loss_formula(self.alpha, self.eps_alpha, d_kl)
        elif self.policy_distribution_str == 'Normal':
            d_kl_mean, d_kl_sigma = self._kl_divergence_normal(policy_old, policy)
            alpha_loss_mu = self._alpha_loss_formula(self.alpha_mu, self.eps_mu, d_kl_mean)
            alpha_loss_sigma = self._alpha_loss_formula(self.alpha_sigma, self.eps_sigma, d_kl_sigma)
            # I believe it is ok to just add these two losses
            alpha_loss = alpha_loss_mu + alpha_loss_sigma
            d_kl = d_kl_mean + d_kl_sigma
        else:
            raise ValueError
        return alpha_loss.mean(), d_kl.mean()

    def _main(self, observations, policy_old, policy, values, actions, returns, advantage):
        # 1) select top-half of advantages and corresponding indices
        row, col, half = self._select_top_adv(advantage)

        # 2) softmax(advantage) with log-sum-exp & policy loss
        max_adv = advantage.max()
        advantage -= max_adv
        softmax_numerator = torch.exp(advantage[row, col] / self.eta)
        softmax_denominator = softmax_numerator.sum()
        softmax_adv = softmax_numerator / softmax_denominator
        policy_loss = self._policy_loss(policy[row, col], actions[row, col], softmax_adv)

        # 3) eta & alpha losses
        log = max_adv + torch.log(softmax_denominator) - np.log(half)
        eta_loss = self.eta * (self.eps_eta + log)
        alpha_loss, d_kl = self._alpha_loss(policy_old, policy)

        # 4) value & full losses
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        loss = value_loss + policy_loss + eta_loss + alpha_loss

        # 5) image_aug loss
        if self.image_augmentation_loss:
            policy_div, value_div = self._augmentation_loss(
                policy.detach(), values.detach(), observations
            )
            loss += policy_div + value_div
            upd = {
                'policy_div': policy_div.item(),
                'value_div': value_div.item()
            }

        grad_norm = self._optimize_loss(loss)

        result = {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'eta_loss': eta_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'eta': self.eta.item(),
            'd_kl': d_kl.item(),
            'loss': loss.item(),
            'grad_norm': grad_norm
        }
        if self.image_augmentation_loss:
            # noinspection PyUnboundLocalVariable
            result.update(upd)

        if self.policy_distribution_str == 'Categorical':
            result['alpha'] = self.alpha.item()
        elif self.policy_distribution_str == 'Normal':
            result['alpha_mu'] = self.alpha_mu.item()
            result['alpha_sigma'] = self.alpha_sigma.item()

        return result

    def _update_eta_alpha(self):
        self.eta.data = torch.max(self.eta, self.param_min)
        if self.policy_distribution_str == 'Categorical':
            self.alpha.data = torch.max(self.alpha, self.param_min)
        elif self.policy_distribution_str == 'Normal':
            self.alpha_mu.data = torch.max(self.alpha_mu, self.param_min)
            self.alpha_sigma.data = torch.max(self.alpha_sigma, self.param_min)

    def _optimize_loss(self, loss):
        self.param_opt.zero_grad()
        grad_norm = super()._optimize_loss(loss)
        self.param_opt.step()
        self._update_eta_alpha()
        return grad_norm

    def _train_fn(self, rollout):
        observations, actions, rewards, not_done, policy_old = self._rollout_to_tensors(rollout)
        policy_old = policy_old.squeeze(1)
        policy, values, returns, advantage = self._compute_returns(observations, rewards, not_done)
        result = self._main(observations, policy_old, policy, values, actions, returns, advantage)
        return result
