import numpy as np


class ExperienceReplay:
    # super-simple and dummy exp_replay
    # can be used to store anything, including whole rollouts
    def __init__(self, capacity):
        self._capacity = capacity
        self._cursor = 0
        self._memory = []

    def __len__(self):
        return len(self._memory)

    def push(self, transition):
        if self._cursor < self._capacity:
            self._memory.append(transition)
        else:
            self._memory[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self._capacity

    def sample(self, batch_size):
        indices = np.random.randint(0, self.__len__() - 1, size=batch_size)
        return [self._memory[i] for i in indices]
