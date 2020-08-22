import os
import time

import json
import gym


def create_log_dir(args):
    # tensorboard logs saved in 'log_dir/tb/', checkpoints in 'log_dir/checkpoints'
    try:
        os.mkdir(args.log_dir)
        os.mkdir(args.log_dir + 'tb_logs')
        os.mkdir(args.log_dir + 'checkpoints')
    except FileExistsError:
        print('log_dir already exists')

    observation_size, action_size, image_env = _parse_env(args.env_name)
    # save training config
    if os.path.exists(args.log_dir + 'config.json'):
        raise FileExistsError('config already exists')
    with open(args.log_dir + 'config.json', 'w') as f:
        args_dict = vars(args)
        args_dict['observation_size'] = observation_size
        args_dict['action_size'] = action_size
        args_dict['image_env'] = image_env
        json.dump(args_dict, f, indent=4)

    return observation_size, action_size, image_env


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper


def _parse_env(env_name):
    env = gym.make(env_name)
    image_env = False
    if len(env.observation_space.shape) == 3:
        image_env = True

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    else:
        action_size = env.action_space.shape[0]

    observation_size = 1
    for i in env.observation_space.shape:
        observation_size *= i

    env.close()
    return observation_size, action_size, image_env
