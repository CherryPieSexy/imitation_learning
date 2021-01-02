import gym


class MarioWrapper(gym.Wrapper):
    # TODO: class docstring
    def __init__(self, env):
        super().__init__(env)

        self.current_lives = 0
        self.current_score = 0

        self.need_reset = True

        self.no_op = [0] * 9

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward /= 10.0

        if info['lives'] < self.current_lives:
            done = True
            reward -= 50

        if info['lives'] == -1:
            self.need_reset = True
        else:
            self.need_reset = False

        # score is already divided by 10,
        # so agent will get +10 for killing an enemy
        # or +10 for pick up a coin
        # or +100 for pick up a mushroom
        reward += (info['score'] - self.current_score)

        self.current_lives = info['lives']
        self.current_score = info['score']

        return observation, reward, done, info

    def reset(self):
        if self.need_reset:
            observation = self.env.reset()

            self.current_lives = 2
            self.current_score = 0

            self.need_reset = False
        else:
            # agent just died, environment shows screen with remaining lives,
            # which is not relevant to training
            for i in range(200):
                observation = self.step(self.no_op)[0]

        # noinspection PyUnboundLocalVariable
        return observation
