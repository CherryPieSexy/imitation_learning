import torch.nn as nn


# In future we will use CNN and RNN, this should be indicated by class name
class ActorCriticMLP(nn.Module):
    # just 3-layer MLP with relu activation and policy & value heads
    def __init__(self, observation_size, action_size, hidden_size):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observation):
        f = self.fe(observation)
        return self.policy(f), self.value(f).squeeze(-1)


class ActorCriticCNN(nn.Module):
    # TODO: implement
    def __init__(self):
        super().__init__()
        # TODO: start with the simplest possible architecture
        pass

    def forward(self, observation):
        pass


class QNetwork(nn.Module):
    # 3-layer MLP with dueling architecture
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
