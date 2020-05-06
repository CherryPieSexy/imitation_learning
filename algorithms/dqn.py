import torch
import torch.nn as nn


from algorithms.policies import Categorical


class NN(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
        )
        self.value_layer = nn.Linear(hidden_size, 1)
        self.advantage_layer = nn.Linear(hidden_size, action_size)

    def forward(self, observation):
        f = self.fe(observation)
        v = self.value_layer(f)
        a = self.advantage_layer(f)
        q = v - (a - a.mean(-1, keepdim=True))
        return q


class DQN:
    def __init__(self,
                 observation_size, action_size, hidden_size,
                 device, lr, gamma):
        self.device = device
        self.nn = NN(observation_size, action_size, hidden_size)
        self.nn.to(device)
        self.target_nn = NN(observation_size, action_size, hidden_size)
        self.nn.to(device)
        self.opt = torch.optim.Adam(self.nn.parameters(), lr)

        self.distribution = Categorical()

        self.gamma = gamma

    def update_target(self):
        pass

    def soft_update_target(self):
        pass

    def act(self, observations, temperature):
        with torch.no_grad():
            q_values = self.nn(
                torch.tensor(observations, dtype=torch.float32)
            )
        action = self.distribution.sample(q_values / temperature, False)
        # return action
        # action = q_values.argmax(-1)
        return action.cpu().numpy()

    def _one_step_target_q(self, q_values, next_q_values, rewards, not_done):
        # like in double Q-Learning
        a_star = q_values.argmax(-1, keepdim=True)[1:]
        next_q_values_for_actions = torch.gather(
            next_q_values, -1,
            a_star
        ).squeeze(-1)
        target_q_value = rewards + self.gamma * not_done * next_q_values_for_actions
        return target_q_value

    def _q_value_loss(self, observations, actions, rewards, not_done):
        q_values = self.nn(observations)  # [T + 1, B, num_actions]
        with torch.no_grad():
            next_q_values = self.nn(observations[1:])
        q_values_for_actions = torch.gather(
            q_values[:-1], -1,
            actions
        ).squeeze(-1)
        target_q_values = self._one_step_target_q(q_values, next_q_values, rewards, not_done)
        q_value_loss = 0.5 * ((q_values_for_actions - target_q_values) ** 2)
        return q_value_loss

    def loss_on_rollout(self, rollout):
        def to_tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        observations, actions, rewards, is_done = list(map(to_tensor, rollout))
        actions = actions.to(torch.long).unsqueeze(-1)
        not_done = 1.0 - is_done

        q_value_loss = self._q_value_loss(observations, actions, rewards, not_done)

        self.opt.zero_grad()
        loss = q_value_loss.mean()
        loss.backward()
        self.opt.step()
