import pickle
import argparse

import json
import torch
import numpy as np
from tqdm import tqdm, trange

from utils.init_env import init_env
from algorithms.nn import ActorCriticTwoMLP, ActorCriticCNN, ActorCriticDeepCNN
from algorithms.agents.policy_gradient import AgentInference


def _to_infinity():
    i = 0
    while True:
        yield i
        i += 1


def play_episode(
        env, agent,
        deterministic, silent, pause
):
    episode_reward, episode_len = 0.0, 0.
    observations, actions, rewards = [], [], []

    obs, done = env.reset(), False
    observations.append(obs)

    if not silent:
        env.render()
        if pause:  # useful to start 'Kazam', select window and record video
            input("press 'enter' to continue...")
    while not done:
        action = agent.act(obs, deterministic=deterministic)
        obs, reward, done, _ = env.step(action, render=not silent)
        episode_reward += reward
        episode_len += 1

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

    if not silent:
        env.render()

    episode = (observations[:-1], actions, rewards)

    return episode_reward, episode_len, episode


def play_n_episodes(
        env, agent,
        deterministic,
        n_episodes, silent,
        reward_threshold, save_demo,
        pause_first
):
    # if 'reward_threshold' is not None, then this function
    # will save #'n_episodes' with episode_reward > reward_threshold

    episode_rewards, episode_lengths = [], []
    episodes_to_save, save_ep_reward = [], []
    total_episodes = 0

    if reward_threshold is not None:
        p_bar = _to_infinity()
        real_p_bar = tqdm(total=n_episodes, ncols=60)
        silent = True
    else:
        if not silent:
            p_bar = range(n_episodes)
        else:
            p_bar = trange(n_episodes, ncols=60)

    for i in p_bar:
        episode_reward, episode_len, episode = play_episode(
            env, agent, deterministic, silent,
            i == 0 and pause_first
        )
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(episode_len)
        total_episodes += 1

        if reward_threshold is not None:
            if episode_reward > reward_threshold:
                episodes_to_save.append(episode)
                save_ep_reward.append(episode_reward)
                # noinspection PyUnboundLocalVariable
                real_p_bar.update()
                if len(episodes_to_save) == n_episodes:
                    real_p_bar.close()
                    with open(save_demo, 'wb') as f:
                        pickle.dump(episodes_to_save, f)

                    print(
                        f'done! '
                        f'Saved {len(episodes_to_save)} episodes with mean reward {np.mean(save_ep_reward)} '
                        f'out of {total_episodes} with mean reward {np.mean(episode_rewards)}'
                    )
                    break

        if not silent:
            print(f'episode_{i} done, len = {episode_len}, reward = {episode_reward}')

    print(f'mean(reward) = {np.mean(episode_rewards)}, std(reward) = {np.std(episode_rewards)}')

    # only for Humanoid:
    # num_fails = sum([1 for i in episode_lengths if i < 1000])
    # max_rewards = [episode_rewards[i] for i in range(n_episodes) if episode_lengths[i] == 1000]
    # print(f'num_fails: {num_fails}, mean_full_reward: {sum(max_rewards) / len(max_rewards)}')


def play_from_folder(
        log_folder, checkpoint_id, deep,
        deterministic, n_episodes, silent,
        reward_threshold, save_demo,
        pause_first
):
    with open(log_folder + 'config.json') as f:
        config = json.load(f)

    action_repeat = config['action_repeat']
    env_name = config['env_name']
    image_env = config['image_env']
    observation_size, action_size = config['observation_size'], config['action_size']
    hidden_size = config['hidden_size']
    policy = config['policy']
    if policy == 'RealNVP':
        policy_args = config['policy_args']
        policy_args = {
            'action_size': action_size,
            'hidden_size': policy_args[0],
            'n_layers': policy_args[1],
            'std_scale': policy_args[2]
        }
    else:
        policy_args = {}

    # initialize agent and environment
    if image_env:
        if deep:
            nn = ActorCriticDeepCNN(action_size, policy)
        else:
            nn = ActorCriticCNN(action_size, policy)
    else:
        nn = ActorCriticTwoMLP(observation_size, action_size, hidden_size, policy)
    device = torch.device('cpu')
    agent = AgentInference(nn, device, policy, policy_args, testing=True)
    agent.eval()
    env = init_env(env_name, 1, action_repeat=action_repeat)
    agent.load(log_folder + f'checkpoints/epoch_{checkpoint_id}.pth')
    agent.eval()
    play_n_episodes(
        env, agent,
        deterministic, n_episodes, silent,
        reward_threshold, save_demo,
        pause_first
    )
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--log_folder', '-l',
        help='log folder that contains checkpoint to load'
    )
    parser.add_argument(
        '--checkpoint_id', '-i',
        help='if of checkpoint to load'
    )
    parser.add_argument(
        '--random', '-r',
        help='if True then action will be sampled from the policy instead from taking mean, default False',
        action='store_true'
    )
    parser.add_argument(
        '--silent', '-s',
        help='if True then episodes will not be shown in window, '
             'and only mean reward will be printed at the end, default False',
        action='store_true'
    )
    parser.add_argument(
        '--pause_first', '-p',
        help='if True, pauses the first episode at the first frame until enter press. '
             'It is useful to record video with Kazam or something else, default False',
        action='store_true'
    )
    parser.add_argument(
        '--n_episodes', '-n',
        help='number of episodes to play or save demo, default 5',
        default=5, type=int
    )
    parser.add_argument(
        '--save_gif', '-g',
        help='file name to save gif of played episodes (max 5) into, not yet implemented',
        default=None, type=str, required=False
    )
    parser.add_argument(
        '--reward_threshold', '-t',
        help='if \'save_demo\' arg provided, then '
             'only episodes with reward > \'reward_threshold\' will be saved into buffer',
        default=None, type=float, required=False
    )
    parser.add_argument(
        '--save_demo', '-d',
        help='file name to save demo of episodes with reward > \'reward_threshold\' into',
        default=None, type=str, required=False
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    play_from_folder(
        args.log_folder, args.checkpoint_id, True,
        not args.random, args.n_episodes, args.silent,
        args.reward_threshold, args.save_demo,
        args.pause_first
    )
