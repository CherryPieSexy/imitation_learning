import gym
import torch
from tqdm import tqdm

from simple.a2c import A2C


agent = A2C(
    3, 1, 64, 'cpu', 'NormalFixed',
    1e-3, 0.99, 0.0
)


def make_env():
    env = gym.make('Pendulum-v0')
    return env


class EnvPool:
    def __init__(self, n_envs):
        self.environments = [make_env() for _ in range(n_envs)]

    def reset(self):
        return [env.reset() for env in self.environments]

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.environments, actions)]
        observation, reward, done, _ = map(list, zip(*results))

        for i in range(len(self.environments)):
            if done[i]:
                observation[i] = self.environments[i].reset()

        return observation, reward, done


env_pool = EnvPool(4)
gamma = 0.99


def gather_rollout(obs, rollout_len):
    obs_, acts_, rews_, is_done_ = [obs], [], [], []
    for i in range(rollout_len):
        act_ = agent.act(obs)
        obs, rew, don = env_pool.step(4.0 * act_ - 2.0)

        obs_.append(obs)
        acts_.append(act_)
        rews_.append(rew)
        is_done_.append(don)
    return obs, (obs_, acts_, rews_, is_done_)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


observations = env_pool.reset()
for step in tqdm(range(50_000)):
    observations, rollout = gather_rollout(observations, 5)
    agent.loss_on_rollout(rollout)


environment = gym.make('Pendulum-v0')
environment = environment


def play_episode(render=False):
    observation = environment.reset()
    rewards = []
    done = False
    i = 0
    while not done:
        action = agent.act([observation])
        i += 1
        observation, reward, done, _ = environment.step(4.0 * action[0] - 2.0)
        if render:
            environment.render()
            # sleep(0.01)

        rewards.append(reward)
    return rewards


for _ in range(10):
    r = play_episode(True)
    print(sum(r))
