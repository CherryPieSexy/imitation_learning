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
        self._num_transitions = self._observations.shape[0]

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
            good_transitions.append([1] * (len(actions) - 1) + [0])

        return \
            np.concatenate(read_observations),\
            np.concatenate(read_actions),\
            np.concatenate(read_rewards),\
            np.concatenate(good_transitions)

    def sample(self):
        # mimics '__iter__' method: every sample will be sampled once per epoch
        if self._pointer + self._batch_size > self._num_transitions:
            np.random.shuffle(self._indices)
            self._pointer = 0

        return {
            'observations': self._observations[self._pointer:self._pointer + self._batch_size],
            'next_observations': self._observations[self._pointer + 1:self._pointer + self._batch_size + 1],
            'actions': self._actions[self._pointer:self._pointer + self._batch_size],
            'rewards': self._rewards[self._pointer:self._pointer + self._batch_size],
            'mask': self._good_transitions[self._pointer:self._pointer + self._batch_size]
        }
