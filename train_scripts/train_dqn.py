import gym
import torch
from time import sleep
from tqdm import tqdm

from simple.dqn import DQN


agent = DQN(
    4, 2, 64, 'cpu', 1e-3, 0.99
)


def make_env():
    env = gym.make('CartPole-v0')
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


def gather_rollout(obs, rollout_len, epsilon):
    # epsilon = prob of agent action
    obs_, acts_, rews_, is_done_ = [obs], [], [], []
    for i in range(rollout_len):
        agent_action = agent.act(obs, epsilon)
        # rand_action = np.random.randint(2, size=(4,))
        # mask_action = np.random.choice(2, size=(4,), p=[epsilon, 1-epsilon])  # 1 if random, 0 if agent
        # env_action = mask_action * rand_action + (1 - mask_action) * agent_action
        env_action = agent_action

        obs, rew, don = env_pool.step(env_action)

        obs_.append(obs)
        acts_.append(env_action)
        rews_.append(rew)
        is_done_.append(don)
    return obs, (obs_, acts_, rews_, is_done_)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


observations = env_pool.reset()
epsilon = 1.0
delta = 0.99 / 4000
for step in tqdm(range(10_000)):
    epsilon = max(epsilon - delta, 0.01)
    observations, rollout = gather_rollout(observations, 5, epsilon)
    agent.loss_on_rollout(rollout)


environment = gym.make('CartPole-v0')
environment = environment.env


def play_episode(render=False):
    observation = environment.reset()
    rewards = []
    done = False
    while not done:
        action = agent.act([observation], 0.01)
        observation, reward, done, _ = environment.step(action[0])
        if render:
            environment.render()
            sleep(0.01)

        rewards.append(reward)
    return rewards


for _ in range(10):
    r = play_episode(True)
    print(sum(r))
