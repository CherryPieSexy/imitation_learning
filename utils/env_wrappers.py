import gym


class ContinuousActionWrapper(gym.Wrapper):
    # rescales agent actions from [-1, +1] to [low, high]
    def __init__(self, env):
        super().__init__(env)
        low = self.action_space.low
        high = self.action_space.high
        self.a = (high - low) / 2.0
        self.b = (high + low) / 2.0

    def step(self, action):
        # noinspection PyTypeChecker
        env_action = self.a * action + self.b
        return self.env.step(env_action)


class OneHotWrapper(gym.Wrapper):
    # transforms one-hot encoded actions to id
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # noinspection PyUnresolvedReferences
        index = action.argmax()
        step_result = self.env.step(index)
        return step_result
