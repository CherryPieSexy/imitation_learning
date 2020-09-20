import pickle
import argparse

import yaml
import torch
import numpy as np
from tqdm import tqdm, trange

from utils.init_env import init_env
from algorithms.nn.actor_critic import init_actor_critic
from algorithms.agents.base_agent import AgentInference


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
        obs, reward, done, info = env.step(action, render=not silent)
        episode_reward += reward
        episode_len += 1

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        # if info['flag_get']:
        #     from time import sleep
        #     sleep(100500)
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
        folder, config_path, checkpoint_path,
        deterministic, silent, pause_first, n_episodes,
        save_gif, reward_threshold, save_demo,
):
    if save_gif:
        raise ValueError('gif saving is not yet implemented...')

    with open(folder + config_path) as f:
        config = yaml.safe_load(f)

    test_env_args = config['test_env_args']
    test_env_args['env_num'] = 1
    test_env = init_env(**test_env_args)

    device = torch.device('cpu')
    nn_online = init_actor_critic(config['actor_critic_nn_type'], config['actor_critic_nn_args'])
    nn_online.to(device)
    policy = config['policy']
    policy_args = config['policy_args']
    agent = AgentInference(nn_online, device, policy, policy_args, testing=True)
    agent.load(folder + checkpoint_path)
    agent.eval()
    play_n_episodes(
        test_env, agent,
        deterministic, n_episodes, silent,
        reward_threshold, save_demo,
        pause_first
    )
    test_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    # config + checkpoint part
    parser.add_argument(
        '--folder', '-f',
        help='this will be added before config and checkpoint paths, default \'\'',
        default=''
    )
    parser.add_argument(
        '--config', '-c',
        help='path to config which contains agent and environment parameters, default \'config.yaml\'',
        default='config.yaml'
    )
    parser.add_argument(
        '--checkpoint', '-p',
        help='path to checkpoint which contains agent weights'
    )

    # playing episodes part
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
        '--pause_first',
        help='if True, pauses the first episode at the first frame until enter press. '
             'It is useful to record video with Kazam or something else, default False',
        action='store_true'
    )
    parser.add_argument(
        '--n_episodes', '-n',
        help='number of episodes to play or save demo, default 5',
        default=5, type=int
    )

    # saving results part
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
        args.folder, args.config, args.checkpoint,
        not args.random, args.silent, args.pause_first, args.n_episodes,
        args.save_gif, args.reward_threshold, args.save_demo,
    )
