import torch

from utils.init_env import init_env
from algorithms.policy_gradient import PolicyGradientInference


def play_episode(env, agent):
    episode_reward = 0.0
    obs, done = env.reset(), False
    while not done:
        env.render()
        action = agent.act(obs, True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


if __name__ == '__main__':
    device = torch.device('cpu')
    agent = PolicyGradientInference(
        False, 24, 4, 64, device, 'Beta'
    )
    agent.eval()
    env = init_env('BipedalWalker-v3', 1)
    agent.load(f'logs/Bipedal/new/22/checkpoints/epoch_0.pth')
    play_episode(env, agent)

    # env.close()
