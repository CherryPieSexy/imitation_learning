import gym

from utils.env_wrappers import RolloutPadWrapper
from utils.vec_env import SubprocVecEnv


class FakeEasyEnv:
    def __init__(self):
        self.state = None
        self._max_episode_steps = 20

    def reset(self):
        self.state = 0

    def step(self, *args, **kwargs):
        self.state += 1
        reward = self.state
        done_ = False
        info_ = {}
        return self.state, reward, done_, info_


def init_env():
    env = gym.make(env_name)
    env = RolloutPadWrapper(env, rollout_len)
    return env


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    rollout_len = 10

    vec_env = SubprocVecEnv([init_env for _ in range(2)])
    vec_env.reset()
    for i in range(100):
        action = [0] * 5
        _, _, done, info = vec_env.step(action)
        print(i, done, info)
    vec_env.close()
