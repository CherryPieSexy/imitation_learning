class MarioWrapper:
    def __init__(self):
        self.prev_lives = 2
        self.prev_score = 0

    def step(self, observation, reward, done, info):
        if info['lives'] < self.prev_lives:
            done = True
            reward -= 50

        if info.get('flag_get', None):
            done = True
            reward += 100

        reward += (info['score'] - self.prev_score) / 2.0
        self.prev_score = info['score']

        return observation, reward / 10.0, done, info

    def reset(self, observation):
        self.prev_lives = 2
        self.prev_score = 0
        return observation
