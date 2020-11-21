import importlib
from _collections import deque

import gym
import cv2
import numpy as np


gym.logger.set_level(40)


def continuous_action_wrapper(env_instance):
    # rescales agent actions from [-1, +1] to [low, high]
    original_step = env_instance.step

    low = env_instance.action_space.low
    high = env_instance.action_space.high
    env_instance.a = (high - low) / 2.0
    env_instance.b = (high + low) / 2.0

    def step(self, action):
        action = np.clip(action, -1.0, +1.0)
        env_action = self.a * action + self.b
        return original_step(env_action)

    env_instance.step = step.__get__(env_instance)
    return env_instance


def one_hot_wrapper(env_instance):
    original_step = env_instance.step

    def step(self, action, **kwargs):
        index = action.argmax()
        return original_step(index, **kwargs)

    env_instance.step = step.__get__(env_instance)
    return env_instance


def action_repeat_wrapper(env_instance, n_repeat):
    original_step = env_instance.step

    def step(self, action, render=False, **kwargs):
        reward = 0.0
        for _ in range(self.n_repeat):
            obs, r, done, info = original_step(action, **kwargs)
            if render:
                env_instance.render()
            reward += r
            if done:
                break
        return obs, reward, done, info

    env_instance.n_repeat = n_repeat
    env_instance.step = step.__get__(env_instance)
    return env_instance


# TODO
def die_penalty_wrapper(env_instance, *args, **kwargs):
    pass
# class DiePenaltyWrapper(gym.Wrapper):
#     def __init__(self, env, penalty=-100):
#         super().__init__(env)
#         self._penalty = penalty
#
#     def step(self, action, **kwargs):
#         observation, reward, done, info = self.env.step(action, **kwargs)
#         if done:
#             if info.get('TimeLimit.truncated', None) is None:
#                 reward += self._penalty
#         return observation, reward, done, info


def image_wrapper(
        env_instance,
        convert_to_gray=True,
        x_start=0, x_end=-1,
        y_start=0, y_end=-1,
        x_size=256, y_size=256
):
    original_step = env_instance.step
    original_reset = env_instance.reset

    def _rgb_2_gray(img):
        img = np.dot(img[..., :], [0.299, 0.587, 0.114])
        img = img / 128.0 - 1
        return img

    def _process_img(img):
        img = img[x_start:x_end, y_start, y_end]
        if convert_to_gray:
            img = _rgb_2_gray(img)
        img = cv2.resize(img, x_size, y_size)
        if convert_to_gray:
            img = img[..., None]
        else:
            img = img / 128.0 - 1
        img = np.transpose(img, (2, 0, 1))
        return img

    def step(self, action, **kwargs):
        observation, reward, done, info = original_step(action, **kwargs)
        observation = _process_img(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = original_reset(**kwargs)
        return _process_img(observation)

    env_instance.step = step.__get__(env_instance)
    env_instance.reset = reset.__get__(env_instance)
    return env_instance


def frame_stack_wrapper(env_instance, n_stack):
    original_step = env_instance.step
    original_reset = env_instance.reset

    env_instance.stack = None
    observation_space = env_instance.observation_space
    if type(observation_space) is gym.spaces.Dict:
        observation_shape = observation_space['observation'].shape
    else:
        observation_shape = observation_space.shape
    new_observation_shape = (observation_shape[0] * n_stack,) + observation_shape[1:]
    new_observation_space = gym.spaces.Box(low=-1, high=+1, shape=new_observation_shape)
    if type(observation_space) is gym.spaces.Dict:
        env_instance.observation_space.spaces['observation'] = new_observation_space
    else:
        env_instance.observation_space = new_observation_space

    def _stack_obs(self, observation):
        if type(observation) is dict:
            self.stack.append(observation['observation'])
        else:
            self.stack.append(observation)
        obs = np.concatenate(np.copy(self.stack), axis=0)
        if type(observation) is dict:
            observation['observation'] = obs
        else:
            observation = obs
        return observation

    def step(self, action, **kwargs):
        observation, reward, done, info = original_step(action, **kwargs)
        self.stack.popleft()
        observation = self._stack_obs(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = original_reset(**kwargs)
        if type(observation) is dict:
            obs = observation['observation']
        else:
            obs = observation
        self.stack = deque([obs] * n_stack)
        obs = np.concatenate(np.copy(self.stack), axis=0)
        if type(observation) is dict:
            observation['observation'] = obs
        else:
            observation = obs
        return observation

    env_instance.step = step.__get__(env_instance)
    env_instance.reset = reset.__get__(env_instance)
    env_instance._stack_obs = _stack_obs.__get__(env_instance)
    return env_instance


# TODO
def state_load_wrapper(env_instance):
    pass
# class StateLoadWrapper(gym.Wrapper):
#     # works for mujoco environments
#     def __init__(self, env, obs_to_state_fn):
#         super().__init__(env)
#         self.obs_to_env_fn = obs_to_state_fn
#
#     def load_state(self, obs):
#         state = self.obs_to_env_fn(obs)
#         self.env.set_state(**state)


def last_achieved_goal_wrapper(env_instance):
    # Stores previously achieved goal and adds it into observation.
    # It is useful because vec_env resets environment immediately after done
    # and last achieved goal (which is contained state after done) is not observable
    original_step = env_instance.step
    original_reset = env_instance.reset

    goal_shape = env_instance.observation_space['achieved_goal'].shape
    fake_goal = np.zeros(goal_shape, dtype=np.float32)
    env_instance.prev_achieved = fake_goal

    def step(self, action, **kwargs):
        observation, reward, done, info = original_step(action, **kwargs)
        observation['prev_achieved_goal'] = self.prev_achieved
        self.prev_achieved = observation['achieved_goal']
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = original_reset(**kwargs)
        observation['prev_achieved_goal'] = self.prev_achieved
        self.prev_achieved = observation['achieved_goal']
        return observation

    env_instance.step = step.__get__(env_instance)
    env_instance.reset = reset.__get__(env_instance)
    return env_instance


def custom_wrapper(env, custom_wrapper_path, custom_wrapper_args):
    """Wraps environment with any wrapper outside this file

    :param env:
    :param custom_wrapper_path: path to function which will be applied after action execution.
           Must have form 'folder.sub_folder.file:class_name'
    :param custom_wrapper_args: arguments of custom wrapper
    """
    module_import_path, class_name = custom_wrapper_path.split(':')
    module = importlib.import_module(module_import_path)
    return getattr(module, class_name)(env, **custom_wrapper_args)
