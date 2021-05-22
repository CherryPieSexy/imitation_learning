import torch.nn as nn

from torch_rl.algorithms.nn.actor_critic import init, actor_critic_forward


class GRUEncoder(nn.Module):
    """
    Simple GRU encoder. Remember: time first!

    If someone want LSTM instead of GRU, encoder must return
    concatenated hidden state and split it before calling forward.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        # basically same initialization as in ikostrikov's repo.
        self.gru = nn.GRU(input_size, output_size)
        self.recurrent = True
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def forward(self, observation, hidden_state):
        without_time = False
        if observation.dim() == 2:  # it means that time size is equal to 1.
            without_time = True
            observation = observation.unsqueeze(0)
        gru_out, hidden_state = self.gru(observation, hidden_state)
        if without_time:
            gru_out = gru_out.squeeze(0)
        return gru_out, hidden_state

    @staticmethod
    def reset_memory_by_ids(memory, ids):
        # hidden_state is a tensor of shape
        # (num_layers * num_directions = 1, batch, hidden_size)
        memory[:, ids] *= 0
        return memory


class CompositeRnnEncoder(nn.Module):
    def __init__(
            self,
            make_encoder,
            gru_input_size, gru_output_size
    ):
        super().__init__()
        self.encoder = make_encoder()
        self.rnn = GRUEncoder(gru_input_size, gru_output_size)
        self.recurrent = True

    def forward(self, observation, hidden_state):
        encoder_result = self.encoder(observation)
        gru_out, hidden_state = self.rnn(encoder_result, hidden_state)
        return gru_out, hidden_state

    def reset_memory_by_ids(self, memory, ids):
        return self.rnn.reset_memory_by_ids(memory, ids)


class OneLayerActorCritic(nn.Module):
    def __init__(
            self,
            input_size, action_size,
            critic_size=1,
            detach_actor=False, detach_critic=False
    ):
        super().__init__()

        self.actor = init(nn.Linear(input_size, action_size), gain=0.01)
        self.critic = init(nn.Linear(input_size, critic_size))

        self.detach_actor = detach_actor
        self.detach_critic = detach_critic

    def forward(self, observation):
        return actor_critic_forward(self, observation)
