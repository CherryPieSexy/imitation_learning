import importlib
from copy import copy
from collections import deque

import gym
import cv2
import numpy as np


gym.logger.set_level(40)


class ContinuousActionWrapper(gym.Wrapper):
    """
    Rescales agent actions from [low, high] to [-1, +1].
    """
    def __init__(self, env):
        super().__init__(env)
        low = self.action_space.low
        high = self.action_space.high
        self.a = (high - low) / 2.0
        self.b = (high + low) / 2.0

    def step(self, action):
        # noinspection PyTypeChecker
        action = np.clip(action, -1.0, +1.0)
        env_action = self.a * action + self.b
        return self.env.step(env_action)


class OneHotWrapper(gym.Wrapper):
    """
    Transforms one-hot encoded actions to id.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # noinspection PyUnresolvedReferences
        index = action.argmax()
        step_result = self.env.step(index)
        return step_result


class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeats action for several environment steps, same as frame-skip.
    Renders skipped frames.
    """
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action, render=False):
        reward = 0.0
        for _ in range(self.action_repeat):
            obs, r, done, info = self.env.step(action)
            if render:
                self.env.render()
            reward += r
            if done:
                break
        # noinspection PyUnboundLocalVariable
        return obs, reward, done, info


class FrameStackWrapper(gym.Wrapper):
    """
    Stacks several last frames (i.e. observations) into one.
    """
    def __init__(self, env, n_stack):
        super().__init__(env)
        self.n_stack = n_stack
        self.stack = None

        observation_shape = self.env.observation_space.shape
        new_observation_shape = (observation_shape[0] * n_stack,) + observation_shape[1:]
        self.observation_space = gym.spaces.Box(
            low=-1, high=+1,
            shape=new_observation_shape
        )

    def reset(self):
        obs = self.env.reset()
        self.stack = deque([obs] * self.n_stack)
        observation = np.concatenate(np.copy(self.stack), axis=0)
        return observation

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        self.stack.popleft()
        self.stack.append(observation)
        observation = np.concatenate(np.copy(self.stack), axis=0)
        return observation, reward, done, info


class DiePenaltyWrapper(gym.Wrapper):
    """
    Adds penalty if environment send done signal before time-limit.
    This is incorrect in environments with 'done on success' (i.e. BipedalWalker or CarRacing).
    """
    def __init__(self, env, penalty=-100):
        super().__init__(env)
        self._penalty = penalty

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        if done:
            if info.get('TimeLimit.truncated', None) is None:
                reward += self._penalty
        return observation, reward, done, info


class TimeStepWrapper(gym.ObservationWrapper):
    """
    Add time-step info into state.
    """
    def __init__(self, env):
        super().__init__(env)
        self._time_step = 0

    def observation(self, observation):
        if len(observation.shape) == 3:
            observation = {'img': observation}

        if type(observation) is dict:
            observation['time_step'] = self._time_step
        elif observation:
            observation = np.concatenate((observation, [self._time_step]), axis=-1)
        return observation

    def step(self, action):
        self._time_step += 1
        return super().step(action)

    def reset(self, **kwargs):
        self._time_step = 0
        return super().env.reset(**kwargs)


class ImageEnvWrapper(gym.Wrapper):
    """
    Provides basic image operation: crop, resize, convert to gray.
    """
    def __init__(
            self, env,
            convert_to_gray=True,
            x_start=0, x_end=-1,
            y_start=0, y_end=-1,
            x_size=256, y_size=256
    ):
        super().__init__(env)
        self.convert_to_gray = convert_to_gray
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_size = x_size
        self.y_size = y_size

        shape = (1 if convert_to_gray else 3, x_size, y_size)
        self.observation_space = gym.spaces.Box(-1, +1, shape=shape)

    @staticmethod
    def _img_2_gray(img):
        img = np.dot(img[..., :], [0.299, 0.587, 0.114])
        img = img / 128.0 - 1
        return img

    def _process_img(self, img):
        img = img[self.x_start:self.x_end, self.y_start:self.y_end]
        if self.convert_to_gray:
            img = self._img_2_gray(img)
        img = cv2.resize(img, (self.x_size, self.y_size))
        if self.convert_to_gray:
            img = img[..., None]  # add channels dim back
        img = np.transpose(img, (2, 0, 1))  # make channels first dimension
        return img

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation = self._process_img(observation)
        return observation, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._process_img(obs)
        return obs


class StateLoadWrapper(gym.Wrapper):
    """
    Works with mujoco environments, useful for resetting environment in some particular state.
    """
    # works for mujoco environments
    def __init__(self, env, obs_to_state_fn):
        super().__init__(env)
        self.obs_to_env_fn = obs_to_state_fn

    def load_state(self, obs):
        state = self.obs_to_env_fn(obs)
        self.env.set_state(**state)


class LastAchievedGoalWrapper(gym.Wrapper):
    """
    Adds previously achieved goal into observation.
    Useful for 'hindsight' tasks because achieved goal in the terminal state is lost
    with vectorised environment.
    """
    def __init__(self, env):
        super().__init__(env)
        goal_shape = self.env.observation_space['achieved_goal'].shape
        fake_goal = np.zeros(goal_shape, dtype=np.float32)
        self.prev_achieved = fake_goal
        self.observation_space.spaces['prev_achieved_goal'] = copy(
            self.observation_space['achieved_goal']
        )

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation['prev_achieved_goal'] = self.prev_achieved
        self.prev_achieved = observation['achieved_goal']
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation['prev_achieved_goal'] = self.prev_achieved
        self.prev_achieved = observation['achieved_goal']
        return observation


class CustomWrapper(gym.Wrapper):
    """
    Wraps environment with any wrapper outside this file.
    """
    def __init__(self, env, custom_wrapper_path, custom_wrapper_args):
        """
        :param env:
        :param custom_wrapper_path: path to file with custom wrapper class.
               Must be written in form 'folder.sub_folder.file:class_name'
        """
        super().__init__(env)
        module_import_path, class_name = custom_wrapper_path.split(':')
        module = importlib.import_module(module_import_path)
        self.custom_wrapper = getattr(module, class_name)(env, **custom_wrapper_args)

    def step(self, action, **kwargs):
        return self.custom_wrapper.step(action, **kwargs)

    def reset(self):
        return self.custom_wrapper.reset()
