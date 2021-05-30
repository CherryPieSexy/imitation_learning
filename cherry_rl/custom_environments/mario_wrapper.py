import gym
import gzip


class MarioWrapper(gym.Wrapper):
    # TODO: class docstring
    def __init__(self, env, state_path=None):
        super().__init__(env)

        self.default_initial_state = True
        if state_path is not None:
            self.default_initial_state = False
            with gzip.open(state_path, 'rb') as f:
                emulator_state = f.read()
            env.initial_state = emulator_state

        self.current_lives = 0
        self.current_score = 0

        self.need_reset = True

        self.no_op = [0] * 9

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

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
        reward += (info['score'] - self.current_score) / 40.0

        self.current_lives = info['lives']
        self.current_score = info['score']

        return observation, reward / 10.0, done, info

    def reset(self):
        if self.need_reset:
            observation = self.env.reset()

            if self.default_initial_state:
                self.current_lives = 2
                self.current_score = 0
            else:
                ram = self.env.get_ram()
                self.current_lives = ram[0x075a]
                self.current_score = int(''.join(map(str, ram[0x07de:0x07de + 6]))) / 10

            self.need_reset = False
        else:
            # agent just died, environment shows screen with remaining lives,
            # which is not relevant to training
            for i in range(200):
                observation = self.step(self.no_op)[0]

        # noinspection PyUnboundLocalVariable
        return observation
