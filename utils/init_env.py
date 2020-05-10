import gym

from utils.vec_env import SubprocVecEnv
from utils.env_wrappers import ContinuousActionWrapper, OneHotWrapper, FrameSkipWrapper

from utils.vec_normalize import VecNormalize


def init_env(
        env_name, num_env,
        relax_discrete=False,
        frame_skip=1, normalize=False,
):
    # WARNING! Wrapper order __is__ important and __must__ be set up carefully!
    def _init_env():
        _env = gym.make(env_name)
        if isinstance(_env.action_space, gym.spaces.Box):
            _env = ContinuousActionWrapper(_env)  # normalize actions to [-1, +1]

        if relax_discrete:
            # can this conflict with something else? Probably yes...
            _env = OneHotWrapper(_env)

        if frame_skip > 1:
            _env = FrameSkipWrapper(_env, frame_skip)

        return _env

    if num_env > 1:
        env = SubprocVecEnv([_init_env for _ in range(num_env)])
    elif num_env == 1:
        env = _init_env()
    else:
        raise ValueError(f'num_env should be >= 1, got num_env={num_env}')

    if normalize:
        # TODO: learn how to use this VecNormalize.
        # Normalization wrapper can only work with Vectorized environment
        env = VecNormalize(env, gamma=0.99)
        raise ValueError('Normalization wrapper not supported yet')

    return env
