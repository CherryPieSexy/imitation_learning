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


if __name__ == '__main__':
    from utils.vec_env import SubprocVecEnv

    def init_env(env_name_, env_id_):
        def _init_env():
            env_ = gym.make(env_name_)
            return env_
        return _init_env


    # single env:
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    obs = env.reset()

    # vec env:
    venv = SubprocVecEnv([init_env(env_name, i) for i in range(5)])
    print(venv.reset())
    done = False
    while not done:
        _, _, done, _ = venv.step([0] * 5)
        done = any(done)
    venv.close()
