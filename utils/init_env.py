import gym

from utils.vec_env import SubprocVecEnv
from utils.env_wrappers import (
    ContinuousActionWrapper, OneHotWrapper, ActionRepeatWrapper,
    DiePenaltyWrapper, FrameStackWrapper,
    ImageEnvWrapper
)


def init_env(
        env_name, num_env,
        time_limit=True,
        die_penalty=0,
        relax_discrete=False,
        action_repeat=1
):
    # WARNING! Wrapper order __is__ important and __must__ be set up carefully!
    # wrappers works like queue: first applied - first executed
    def _init_env():
        _env = gym.make(env_name)

        if not time_limit:
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
            # _env = ImageEnvWrapper(_env, True, 0, -12, 0, -1, 42, 42)
            _env = ImageEnvWrapper(_env, True, 0, -1, 0, -1, 96, 96)
            _env = FrameStackWrapper(_env, 4)

        return _env

    if num_env > 1:
        env = SubprocVecEnv([_init_env for _ in range(num_env)])
    elif num_env == 1:
        env = _init_env()
    else:
        raise ValueError(f'num_env should be >= 1, got num_env={num_env}')

    return env
