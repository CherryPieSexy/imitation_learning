import torch
from torch.nn.utils import clip_grad_norm_

from algorithms.distributions import distributions_dict
from algorithms.nn import ModelMLP


class EnvModel:
    def __init__(
            self,
            observation_size, action_size, hidden_size, device,
            predict_reward, use_distribution,
            lr, clip_grad
    ):
        self.device = device
        if use_distribution:
            self.distribution = distributions_dict['Normal']()

        self.predict_reward = predict_reward
        self.use_distribution = use_distribution
        self.nn = ModelMLP(
            observation_size, action_size, hidden_size,
            predict_reward, use_distribution
        )
        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)
        self.clip_grad = clip_grad
        self.training = True

    def save(self):
        pass

    def load(self):
        pass

    def train(self):
        self.training = True
        self.nn.eval()

    def eval(self):
        self.training = False
        self.nn.train()

    # technically mse is likelihood too...
    def _mse_loss(self, state, action, next_state):
        prediction = self.nn(state, action)
        delta = prediction - next_state
        loss = 0.5 * (delta ** 2)
        return loss

    def _likelihood_loss(self, observations, action, next_observations):
        prediction = self.nn(observations, action)
        log_prob = self.distribution.log_prob(prediction, next_observations)
        loss = -log_prob  # we want to MAXIMIZE log-prob of true data
        return loss

    def _loss(self, observations, action, next_observations, not_done):
        if self.use_distribution:
            loss = self._likelihood_loss(observations, action, next_observations)
        else:
            loss = self._mse_loss(observations, action, next_observations)
        loss = (not_done * loss).sum() / not_done.sum()
        return loss

    def _optimize_loss(self, loss):
        self.opt.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(
            self.nn.parameters(), self.clip_grad
        )
        self.opt.step()
        return gradient_norm

    def train_on_rollout(self, rollout):
        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        # how to use 'is_done' here? I am not sure...
        # at least we should mask transitions with done from loss
        observations, actions, rewards, is_done = list(map(to_tensor, rollout))
        observations, next_observations = observations[:-1], observations[1:]
        not_done = 1.0 - is_done
        if self.predict_reward:
            next_observations = torch.cat([next_observations, rewards], dim=-1)

        loss = self._loss(observations, actions, next_observations, not_done)
        grad_norm = self._optimize_loss(loss)
        result = {
            'loss': loss.item(),
            'grad_norm': grad_norm
        }
        return result
