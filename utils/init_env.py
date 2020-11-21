import importlib

import gym
import retro

from utils.vec_env import SubprocVecEnv
from utils.env_wrappers import (
    continuous_action_wrapper, one_hot_wrapper, action_repeat_wrapper,
    die_penalty_wrapper, frame_stack_wrapper,
    image_wrapper, last_achieved_goal_wrapper, custom_wrapper
)


def init_env(
        env_type, env_name, env_args,
        env_num,
        add_prev_achieved_goal=False,
        time_unlimited=False,
        die_penalty=0,
        relax_discrete=False,
        action_repeat=1,
        frame_stack=1,
        image_args=None,
        custom_wrapper_path=None,
        custom_wrapper_args=None,
        flatten_observation=False
):
    """Function to init environment.

    WARNING! Wrapper order __is__ important and __must__ be set up carefully!
    wrappers works like queue: first applied - first executed

    :param env_type: gym, retro, or path to any environment in form
                     'folder.sub_folder.file', str
    :param env_name:
    :param env_args:
    :param env_num: number of environments working in parallel
    :param add_prev_achieved_goal: useful for goal-augmented environment and hindsight
    :param time_unlimited:
    :param die_penalty:
    :param relax_discrete:
    :param action_repeat:
    :param frame_stack:
    :param image_args: dict with keys:
                       convert_to_gray,
                       x_start, x_end,
                       y_start, y_end,
                       x_size, y_size
    :param custom_wrapper_path:
    :param custom_wrapper_args:
    :param flatten_observation: if True then multi-part observation will be flattened,
                                useful for goal-augmented environments
    :return: environment instance
    """
    def _init_env():
        if env_type == 'gym':
            maker = gym.make
            env_args['id'] = env_name
        elif env_type == 'retro':
            maker = retro.make
            env_args['game'] = env_name
        else:
            module = importlib.import_module(env_type)
            maker = getattr(module, env_name)

        _env = maker(**env_args)
        if flatten_observation:
            # noinspection PyUnresolvedReferences
            _env = gym.wrappers.FlattenObservation(_env)

        if custom_wrapper_path is not None:
            _env = custom_wrapper(_env, custom_wrapper_path, custom_wrapper_args)

        if add_prev_achieved_goal:
            _env = last_achieved_goal_wrapper(_env)

        if time_unlimited:
            _env = _env.env

        if die_penalty != 0:
            _env = die_penalty_wrapper(_env, die_penalty)

        if isinstance(_env.action_space, gym.spaces.Box):
            _env = continuous_action_wrapper(_env)  # normalize actions to [-1, +1]

        if relax_discrete:
            _env = one_hot_wrapper(_env)

        # this wrapper is useful even if action_repeat=1, because it supports rendering
        _env = action_repeat_wrapper(_env, action_repeat)

        # this is common for image envs
        if _env.observation_space.shape is not None:
            if len(_env.observation_space.shape) == 3:
                _env = image_wrapper(_env, **image_args)
                _env = frame_stack_wrapper(_env, 4 if frame_stack == 1 else frame_stack)
        elif frame_stack != 1:
            _env = frame_stack_wrapper(_env, frame_stack)

        return _env

    if env_num > 1:
        env = SubprocVecEnv([_init_env for _ in range(env_num)])
    elif env_num == 1:
        env = _init_env()
    else:
        raise ValueError(f'num_env should be >= 1, got num_env={env_num}')

    return env
