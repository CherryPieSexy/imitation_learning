import gym
import torch
from tqdm import tqdm

from algorithms.ppo import PPO


env_name = 'Pendulum-v0'

agent = PPO(
    3, 1, 256, 'cpu',
    'Beta', True, 'gae',
    3e-4, 0.99, 1e-3, 100500,
    0.95,
    ppo_epsilon=0.1, ppo_n_epochs=4, ppo_mini_batch=160
)


def make_env():
    env = gym.make(env_name)
    return env


class EnvPool:
    def __init__(self, n_envs):
        self.environments = [make_env() for _ in range(n_envs)]

    def reset(self):
        return [env.reset() for env in self.environments]

    def step(self, actions):
        results = [env.step(2 * a) for env, a in zip(self.environments, actions)]
        observation, reward, done, _ = map(list, zip(*results))

        for i in range(len(self.environments)):
            if done[i]:
                observation[i] = self.environments[i].reset()

        return observation, reward, done


env_pool = EnvPool(16)
gamma = 0.99


def gather_rollout(obs, rollout_len):
    obs_, acts_, rews_, is_done_ = [obs], [], [], []
    for i in range(rollout_len):
        act_ = agent.act(obs)
        obs, rew, don = env_pool.step(act_)

        obs_.append(obs)
        acts_.append(act_)
        rews_.append([r_ for r_ in rew])
        is_done_.append(don)
    return obs, (obs_, acts_, rews_, is_done_)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


observations = env_pool.reset()
for step in tqdm(range(20_000)):
    observations, rollout = gather_rollout(observations, 20)
    agent.train_on_rollout(rollout)


environment = gym.make(env_name)
environment = environment


def play_episode(render=False):
    observation = environment.reset()
    rewards = []
    done = False
    i = 0
    while not done:
        action = agent.act([observation])
        i += 1
        observation, reward, done, _ = environment.step(2 * action[0])
        if render:
            environment.render()
            # sleep(0.01)

        rewards.append(reward)
    return rewards


for _ in range(10):
    r = play_episode(True)
    print(sum(r))
