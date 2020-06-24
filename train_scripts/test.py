import torch
import numpy as np

from utils.init_env import init_env
from algorithms.policy_gradient import PolicyGradientInference


def play_episode():
    episode_reward = 0.0
    obs, done = env.reset(), False
    while not done:
        env.render()
        action = agent.act(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def play_n_episodes():
    episode_rewards = []
    for i in range(n):
        episode_reward = play_episode()
        episode_rewards.append(float(episode_reward))
        print(i, episode_reward)
    print(np.mean(episode_rewards), np.std(episode_rewards))


if __name__ == '__main__':
    device = torch.device('cpu')
    agent = PolicyGradientInference(
        False, 24, 4, 64, device, 'Beta'
    )
    agent.eval()
    env = init_env('BipedalWalker-v3', 1)
    agent.load(f'logs/Bipedal/new/22/checkpoints/epoch_5.pth')
    n = 20
    play_n_episodes()

    env.close()
