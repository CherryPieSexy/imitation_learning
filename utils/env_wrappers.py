import importlib
from _collections import deque

import gym
import cv2
import numpy as np


gym.logger.set_level(40)


class ContinuousActionWrapper(gym.Wrapper):
    # rescales agent actions from [-1, +1] to [low, high]
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
    # transforms one-hot encoded actions to id
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # noinspection PyUnresolvedReferences
        index = action.argmax()
        step_result = self.env.step(index)
        return step_result


class ActionRepeatWrapper(gym.Wrapper):
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


class DiePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-100):
        super().__init__(env)
        self._penalty = penalty

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        if done:
            if info.get('TimeLimit.truncated', None) is None:
                reward += self._penalty
        return observation, reward, done, info


class ImgToGrayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def _img_2_gray(img):
        img = np.dot(img[..., :], [0.299, 0.587, 0.114])
        img = img / 128.0 - 1
        return img

    def reset(self):
        observation = self.env.reset()
        observation = self._img_2_gray(observation)
        return observation

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation = self._img_2_gray(observation)
        return observation, reward, done, info


class ImageEnvWrapper(gym.Wrapper):
    # crop and resize
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

    def reset(self):
        obs = self.env.reset()
        obs = self._process_img(obs)
        return obs

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation = self._process_img(observation)
        return observation, reward, done, info


class FrameStackWrapper(gym.Wrapper):
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


class StateLoadWrapper(gym.Wrapper):
    # works for mujoco environments
    def __init__(self, env, obs_to_state_fn):
        super().__init__(env)
        self.obs_to_env_fn = obs_to_state_fn

    def load_state(self, obs):
        state = self.obs_to_env_fn(obs)
        self.env.set_state(**state)


class CustomWrapper(gym.Wrapper):
    def __init__(self, env, custom_wrapper_path, custom_wrapper_args):
        """Wraps environment with any wrapper outside this file

        :param env:
        :param custom_wrapper_path: path to function which will be applied after action execution.
               Must have form 'folder.sub_folder.file:class_name'
        """
        super().__init__(env)
        module_import_path, class_name = custom_wrapper_path.split(':')
        module = importlib.import_module(module_import_path)
        self.custom_wrapper = getattr(module, class_name)(env, **custom_wrapper_args)

    def step(self, action, **kwargs):
        return self.custom_wrapper.step(action, **kwargs)

    def reset(self):
        return self.custom_wrapper.reset()
