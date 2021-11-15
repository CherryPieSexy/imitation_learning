import pickle
import numpy as np


class TransitionsDemoBuffer:
    """
    Reads demo file recorded by 'test.py' script.
    """
    def __init__(self, filename, batch_size):
        observations, actions, rewards, good_transitions = self._load_from_file(filename)
        self._observations = observations
        self._actions = actions
        self._rewards = rewards
        self._good_transitions = good_transitions
        # I drop the last transition from each episode during reading data,
        # but I use last next transitions during sampling ->
        # it does not exists for the last observation,
        # so I have to reduce number of transitions by 1
        self._num_transitions = self._observations.shape[0] - 1

        self._batch_size = batch_size
        self._indices = np.arange(self._num_transitions)
        np.random.shuffle(self._indices)
        self._pointer = 0

    @staticmethod
    def _load_from_file(filename):
        # demo file = list of episodes
        # episode = [observations, actions, rewards]
        # observations = np.array of shape (time, obs_dim)
        # similar for actions and rewards
        with open(filename, 'rb') as f:
            episodes = pickle.load(f)

        read_observations, read_actions, read_rewards = [], [], []
        good_transitions = []
        for observations, actions, rewards in episodes:
            read_observations.append(observations[:-1])
            read_actions.append(actions)
            read_rewards.append(rewards)
            # last transition is bad, because it has no 'next_observation'
            good_transitions.append([1] * (len(actions) - 1) + [0])

        return \
            np.concatenate(read_observations),\
            np.concatenate(read_actions),\
            np.concatenate(read_rewards),\
            np.concatenate(good_transitions)

    def sample(self):
        # mimics '__iter__' method: every sample will be sampled once per epoch
        if self._pointer + self._batch_size + 1 > self._num_transitions:
            np.random.shuffle(self._indices)
            self._pointer = 0

        selected_indices = self._indices[self._pointer:self._pointer + self._batch_size]
        result = {
            'observations': self._observations[selected_indices],
            'next_observations': self._observations[selected_indices + 1],
            'actions': self._actions[selected_indices],
            'rewards': self._rewards[selected_indices],
            'mask': self._good_transitions[selected_indices]
        }
        self._pointer += self._batch_size
        return result
