import pickle
import argparse
import importlib

import torch
import numpy as np
from tqdm import tqdm, trange


device = torch.device('cpu')


def _to_infinity():
    i = 0
    while True:
        yield i
        i += 1


def _env_action(action):
    if type(action) is tuple:
        env_action = tuple([
            [mode[i][0].cpu().numpy() for mode in action]
            for i in range(len(action[0]))
        ])
    else:
        env_action = action[0].cpu().numpy()
    return env_action


def play_episode(
        env, model,
        deterministic, silent, pause
):
    episode_reward, episode_len = 0.0, 0.
    observations, actions, rewards = [], [], []

    obs, done = env.reset(), False
    memory = None
    observations.append(obs)

    if not silent:
        env.render()
        if pause:  # useful to start 'Kazam', select window and record video
            input("press 'enter' to continue...")
    while not done:
        # agent always takes observation with [batch, *dim(obs)] size as input
        # and returns action and log-prob with corresponding size
        if type(obs) is dict:
            act_obs = {key: value[None, :] for key, value in obs.items()}
        else:
            act_obs = [obs]
        act_result = model.act(act_obs, memory, deterministic=deterministic)
        action = act_result['action']
        memory = act_result['memory']

        env_action = _env_action(action)
        obs, reward, done, info = env.step(env_action, render=not silent)

        episode_reward += reward
        episode_len += 1

        observations.append(obs)
        actions.append(env_action)
        rewards.append(reward)

    episode = (
        np.array(observations[:-1], dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32)
    )

    return episode_reward, episode_len, episode


def play_n_episodes(
        env, model,
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
            env, model, deterministic, silent,
            i == 0 and pause_first
        )
        episode_rewards.append(float(episode_reward))
        episode_lengths.append(episode_len)
        total_episodes += 1

        if reward_threshold is not None:
            if episode_reward >= reward_threshold:
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
                        f'Saved {len(episodes_to_save)} episodes out of {total_episodes};\n'
                        f'mean reward of saved episodes: {np.mean(save_ep_reward)}\n'
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
        folder, checkpoint_id,
        deterministic, silent, pause_first, n_episodes,
        reward_threshold, save_demo,
):
    config = importlib.import_module(folder.replace('/', '.') + 'config')

    test_env = config.make_env()()

    model = config.make_ac_model()
    checkpoint = torch.load(folder + 'checkpoints/' + f'epoch_{checkpoint_id}.pth')
    model.load_state_dict(checkpoint['ac_model'])
    model.to(device)
    model.eval()

    play_n_episodes(
        test_env, model,
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
        help='folder with experiment config and checkpoints',
        type=str
    )
    parser.add_argument(
        '--checkpoint_id', '-p',
        help='index of checkpoint to load',
        type=str
    )

    # playing episodes part
    parser.add_argument(
        '--random', '-r',
        help='if True then action will be sampled from the policy instead from taking mean, default False',
        action='store_true'
    )
    parser.add_argument(
        '--silent', '-s',
        help='if True then environment will not be rendered '
             'and only mean reward will be printed at the end, default False',
        action='store_true'
    )
    parser.add_argument(
        '--pause_first',
        help='if True, pauses the first episode at the first frame until enter pressed. '
             'It is useful to record video with screen capturing program (i.e. Kazam), '
             'default False',
        action='store_true'
    )
    parser.add_argument(
        '--n_episodes', '-n',
        help='number of episodes to play or save demo, default 10',
        default=10, type=int
    )

    # saving results part
    parser.add_argument(
        '--save_demo', '-d',
        help='file name to save demo of episodes with reward > \'reward_threshold\' into',
        default=None, type=str, required=False
    )
    parser.add_argument(
        '--reward_threshold', '-t',
        help='if \'save_demo\' arg provided, then '
             'only episodes with reward > \'reward_threshold\' will be saved into buffer',
        default=None, type=float, required=False
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    play_from_folder(
        args.folder, args.checkpoint_id,
        not args.random, args.silent, args.pause_first, args.n_episodes,
        args.reward_threshold, args.save_demo,
    )
