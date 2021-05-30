import torch


def _one_step_returns(
        gamma,
        value, rewards, not_done
):
    returns = rewards + gamma * not_done * value[1:]
    return returns


def _n_step_returns(
        gamma,
        value, rewards, not_done
):
    rollout_len = rewards.size(0)
    last_value = value[-1]
    returns = []
    for t in reversed(range(rollout_len)):
        last_value = rewards[t] + gamma * not_done[t] * last_value
        returns.append(last_value)
    returns = torch.stack(returns[::-1])
    return returns


def _gae(
        gamma, gae_lambda,
        value, rewards, not_done
):
    rollout_len = rewards.size(0)
    gae = 0
    returns = []
    for t in reversed(range(rollout_len)):
        delta = rewards[t] + gamma * not_done[t] * value[t + 1] - value[t]
        gae = delta + gamma * gae_lambda * not_done[t] * gae
        returns.append(gae + value[t])
    returns = torch.stack(returns[::-1])
    return returns


def _v_trace(
        gamma, gae_lambda,
        value, rewards, not_done,
        old_log_pi, log_pi,
):
    rollout_len = rewards.size(0)
    v_trace = value[-1]
    returns = []
    ratio = (log_pi - old_log_pi).exp()
    ratio = torch.clamp(ratio, 0.05, 20.0)

    for t in reversed(range(rollout_len)):
        rho = torch.clamp_min_(ratio[t], 1.0).unsqueeze(-1)
        c = gae_lambda * torch.clamp_min_(ratio[t], 1.0).unsqueeze(-1)

        delta = rho * (rewards[t] + gamma * not_done[t] * value[t + 1] - value[t])
        v_trace = value[t] + delta + gamma * not_done[t] * c * (v_trace - value[t + 1])
        returns.append(v_trace)
    returns = torch.stack(returns[::-1])
    return returns


class ReturnsEstimator:
    def __init__(self, method, gamma=0.99, gae_lambda=0.9):
        self.method = method
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def policy_value_returns_adv(self, actor_critic_model, data_dict):
        # long but descriptive name.
        observations = data_dict['observations']
        rewards = data_dict['rewards']
        not_done = 1.0 - data_dict['is_done']
        memory = data_dict['memory']

        policy, value, _ = actor_critic_model(observations, memory)
        with torch.no_grad():
            if self.method == '1-step':
                returns = _one_step_returns(self.gamma, value, rewards, not_done)
            elif self.method == 'n-step':
                returns = _n_step_returns(self.gamma, value, rewards, not_done)
            elif self.method == 'gae':
                returns = _gae(self.gamma, self.gae_lambda, value, rewards, not_done)
            elif self.method == 'v-trace':
                log_prob = actor_critic_model.pi_distribution.log_prob(
                    policy[:-1], data_dict['actions']
                )
                returns = _v_trace(
                    self.gamma, self.gae_lambda,
                    value, rewards, not_done,
                    data_dict['log_prob'], log_prob
                )
        policy, value = policy[:-1], value[:-1]
        advantage = (returns - value).sum(-1).detach()
        return policy, value, returns, advantage
