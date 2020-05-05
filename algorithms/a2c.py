import torch
from torch.nn.utils import clip_grad_norm_

from .policies import distributions_dict
from .nn import ActorCriticNN


class A2C:
    """
    can act (with grad, it may be useful) and calculate loss on rollout
    """
    def __init__(
            self,
            observation_size, action_size, hidden_size, device,
            distribution, normalize_adv, returns_estimator,
            lr, gamma, entropy, clip_grad,
            gae_lambda=0.95
    ):
        """

        :param observation_size, action_size, hidden_size, device: just nn parameters
        :param distribution: 'Categorical' or 'NormalFixed' or 'Beta'
        :param normalize_adv: True or False
        :param returns_estimator: '1-step', 'n-step', 'gae'
        :param lr, gamma, entropy, clip_grad: learning hyper-parameters
        :param gae_lambda: gae lambda, optional
        """
        self.device = device

        self.distribution = distributions_dict[distribution]()
        if distribution in ['Beta', 'Normal']:
            action_size *= 2

        # policy, value = nn(obs)
        # policy.size() == (T, B, dim(A))
        # value.size() == (T, B)  - there is no '1' at the last dimension!
        self.nn = ActorCriticNN(observation_size, action_size, hidden_size)
        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)

        self.normalize_adv = normalize_adv
        self.returns_estimator = returns_estimator
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.entropy = entropy
        self.clip_grad = clip_grad

    def act(self, observations):
        """
        :param observations: np.array of observation, shape = [T, B, dim(obs)]
        :return: action, np.array of shape = [T, B, dim(action)]
        """
        logits, _ = self.nn(
            torch.tensor(observations, dtype=torch.float32)
        )
        action = self.distribution.sample(logits, False)
        return action.cpu().numpy()

    def _one_step_returns(self, next_values, rewards, not_done):
        returns = rewards + self.gamma * not_done * next_values
        return returns

    def _n_step_returns(self, next_values, rewards, not_done):
        rollout_len = rewards.size(0)
        last_value = next_values[-1]
        returns = []
        for t in reversed(range(rollout_len)):
            last_value = rewards[t] + self.gamma * not_done[t] * last_value
            returns.append(last_value)
        returns = torch.stack(returns[::-1])
        return returns

    def _gae(self, values, next_values, rewards, not_done):
        rollout_len = rewards.size(0)
        values = torch.cat([values, next_values[-1:]], dim=0)
        gae = 0
        returns = []
        for t in reversed(range(rollout_len)):
            delta = rewards[t] + self.gamma * not_done[t] * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done[t] * gae
            returns.append(gae + values[t])
        returns = torch.stack(returns[::-1])
        return returns

    @staticmethod
    def _normalize_advantage(advantage):
        # advantages normalized across batch dimension, I think this is correct
        mean = advantage.mean(dim=1, keepdim=True)
        std = (advantage.std(dim=1, keepdim=True) + 1e-5)
        advantage = (advantage - mean) / std
        return advantage

    def _policy_loss(self, policy, actions, advantage):
        log_pi_for_actions = self.distribution.log_prob(policy, actions)
        if self.normalize_adv:
            advantage = self._normalize_advantage(advantage)
        policy_loss = log_pi_for_actions * advantage.detach()
        return policy_loss.mean()

    def _optimize_loss(self, loss):
        self.opt.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(
            self.nn.parameters(), self.clip_grad
        )
        self.opt.step()
        return gradient_norm

    def _estimate_returns(self, values, next_values, rewards, not_done):
        with torch.no_grad():  # returns should not have gradients in any case!
            if self.returns_estimator == '1-step':
                returns = self._one_step_returns(next_values, rewards, not_done)
            elif self.returns_estimator == 'n-step':
                returns = self._n_step_returns(next_values, rewards, not_done)
            elif self.returns_estimator == 'gae':
                returns = self._gae(values, next_values, rewards, not_done)
            else:
                raise ValueError('unknown returns estimator')
        return returns.detach()

    def _calc_losses(self, policy, actions, advantage):
        value_loss = 0.5 * (advantage ** 2).mean()
        policy_loss = self._policy_loss(policy, actions, advantage)
        entropy = self.distribution.entropy(policy).mean()
        loss = value_loss - policy_loss - self.entropy * entropy
        return value_loss, policy_loss, entropy, loss

    # ugly *args, **kwargs for inheritance -_-
    def _main(self, *args, **kwargs):
        policy, actions, advantage = args
        # simple for A2C: just call _calc_losses once and optimize them
        # slightly more complex for PPO:
        # it optimizes losses for several steps, but it returns same values
        value_loss, policy_loss, entropy, loss = self._calc_losses(
            policy, actions, advantage
        )
        grad_norm = self._optimize_loss(loss)
        result = (
            value_loss.item(), policy_loss.item(),
            entropy.item(), loss.item(), grad_norm
        )
        return result

    def loss_on_rollout(self, rollout):
        """
        :param rollout: tuple (observations, actions, rewards, is_done),
               where each one is np.array of shape [time, batch, ...] except observations,
               observations of shape [time + 1, batch, ...]
        :return: dict of loss values, gradient norm, mean reward on rollout
        """
        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        observations, actions, rewards, is_done = list(map(to_tensor, rollout))
        policy, values = self.nn(observations)
        policy = policy[:-1]
        values, next_values = values[:-1], values[1:].detach()
        not_done = 1.0 - is_done

        returns = self._estimate_returns(values, next_values, rewards, not_done)
        advantage = returns - values

        value_loss, policy_loss, entropy, loss, grad_norm = self._main(
            policy, actions, advantage
        )

        # PPO has same result_dict
        result = {
            'value_loss': value_loss,
            'policy_loss': policy_loss,
            'entropy': entropy,
            'loss': loss,
            'grad_norm': grad_norm,
            'reward': rewards.mean().item()
        }

        return result
