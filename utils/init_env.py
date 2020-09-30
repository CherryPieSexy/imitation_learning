import importlib

import gym
import retro

from utils.vec_env import SubprocVecEnv
from utils.env_wrappers import (
    ContinuousActionWrapper, OneHotWrapper, ActionRepeatWrapper,
    DiePenaltyWrapper, FrameStackWrapper,
    ImageEnvWrapper, CustomWrapper
)


def init_env(
        env_type, env_name, env_args,
        env_num,
        time_unlimited=False,
        die_penalty=0,
        relax_discrete=False,
        action_repeat=1,
        image_args=None,
        custom_wrapper_path=None,
        custom_wrapper_args=None
):
    """Function to init environment.

    WARNING! Wrapper order __is__ important and __must__ be set up carefully!
    wrappers works like queue: first applied - first executed

    :param env_type: gym, retro, or path to any environment in form
                     'folder.sub_folder.file', str
    :param env_name:
    :param env_args:
    :param env_num: number of environments working in parallel
    :param time_unlimited:
    :param die_penalty:
    :param relax_discrete:
    :param action_repeat:
    :param image_args: dict with keys:
                       convert_to_gray,
                       x_start, x_end,
                       y_start, y_end,
                       x_size, y_size
    :param custom_wrapper_path:
    :param custom_wrapper_args:
    :return:
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

        if custom_wrapper_path is not None:
            _env = CustomWrapper(_env, custom_wrapper_path, custom_wrapper_args)

        if time_unlimited:
            _env = _env.env

        if die_penalty != 0:
            _env = DiePenaltyWrapper(_env, die_penalty)

        if isinstance(_env.action_space, gym.spaces.Box):
            _env = ContinuousActionWrapper(_env)  # normalize actions to [-1, +1]

        if relax_discrete:
            _env = OneHotWrapper(_env)

        # this wrapper is useful even if action_repeat=1, because it supports rendering
        _env = ActionRepeatWrapper(_env, action_repeat)

        # this is common for image envs
        if len(_env.observation_space.shape) == 3:
            _env = ImageEnvWrapper(_env, **image_args)
            _env = FrameStackWrapper(_env, 4)

        return _env

    if env_num > 1:
        env = SubprocVecEnv([_init_env for _ in range(env_num)])
    elif env_num == 1:
        env = _init_env()
    else:
        raise ValueError(f'num_env should be >= 1, got num_env={env_num}')

    return env
