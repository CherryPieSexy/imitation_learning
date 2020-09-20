import os
import time

import gym


def create_log_dir(log_dir):
    # tensorboard logs saved in 'log_dir/tb/', checkpoints in 'log_dir/checkpoints'
    try:
        os.mkdir(log_dir)
        os.mkdir(log_dir + 'tb_logs')
        os.mkdir(log_dir + 'checkpoints')
    except FileExistsError:
        print('log_dir already exists')

    # save training config
    if os.path.exists(log_dir + 'config.yaml'):
        raise FileExistsError('config already exists')


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper


def parse_env(env_name):
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
