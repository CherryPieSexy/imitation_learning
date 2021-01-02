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
        self._a = (high - low) / 2.0
        self._b = (high + low) / 2.0

    def step(self, action):
        # noinspection PyTypeChecker
        action = np.clip(action, -1.0, +1.0)
        env_action = self._a * action + self._b
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


class ActionRepeatAndRenderWrapper(gym.Wrapper):
    """
    Repeats action for several environment steps, same as frame-skip.
    Must be applied for every environment since it do rendering during
    calling step method with render=True kwarg.
    """
    def __init__(self, env, action_repeat=1):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action, render=False):
        reward = 0.0
        for _ in range(self._action_repeat):
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
        self._n_stack = n_stack
        self._stack = None

        observation_shape = self.env.observation_space.shape
        new_observation_shape = (observation_shape[0] * n_stack,) + observation_shape[1:]
        self._observation_space = gym.spaces.Box(
            low=-1, high=+1,
            shape=new_observation_shape
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._stack = deque([obs] * self._n_stack)
        observation = np.concatenate(np.copy(self._stack), axis=0)
        return observation

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        self._stack.popleft()
        self._stack.append(observation)
        observation = np.concatenate(np.copy(self._stack), axis=0)
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
    def __init__(self, env, max_time=1):
        super().__init__(env)
        self._time_step = 0
        self._max_time = max_time

    def observation(self, observation):
        if len(observation.shape) == 3:
            observation = {'img': observation}

        time_info = self._time_step / self._max_time
        if type(observation) is dict:
            observation['time_step'] = time_info
        else:
            observation = np.concatenate((observation, [time_info]), axis=-1)
        return observation

    def step(self, action, **kwargs):
        self._time_step += 1
        observation, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        self._time_step = 0
        return super().reset(**kwargs)


class DeSyncWrapper(gym.Wrapper):
    """
    Wrapper for environment desynchronization.
    For example useful in 'CarRacing' environment where episode segments in
    consecutive rollouts are highly correlated: agent can obtain high reward
    at the beginning of episode (frames until first road bend)
    and can do nothing when it is off road.
    """
    def __init__(self, env, max_time=None, env_id=0, n_envs=1):
        super().__init__(env)
        self._n_steps = env_id * max_time // n_envs
        self._first_reset = True

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        if self._first_reset:
            obs = self.env.reset(**kwargs)
            for _ in range(self._n_steps):
                obs, _, _, _ = self.env.step(self.action_space.sample())
            self._first_reset = False
            return obs
        else:
            return self.env.reset(**kwargs)


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
        self._convert_to_gray = convert_to_gray
        self._x_start = x_start
        self._x_end = x_end
        self._y_start = y_start
        self._y_end = y_end
        self._x_size = x_size
        self._y_size = y_size

        shape = (1 if convert_to_gray else 3, x_size, y_size)
        self.observation_space = gym.spaces.Box(-1, +1, shape=shape)

    @staticmethod
    def _img_2_gray(img):
        img = np.dot(img[..., :], [0.299, 0.587, 0.114])
        img = img / 128.0 - 1
        return img

    def _process_img(self, img):
        img = img[self._x_start:self._x_end, self._y_start:self._y_end]
        if self._convert_to_gray:
            img = self._img_2_gray(img)
        img = cv2.resize(img, (self._x_size, self._y_size))
        if self._convert_to_gray:
            img = img[..., None]  # add channels dim back
        img = np.transpose(img, (2, 0, 1))  # make channels first dimension
        return img

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation = self._process_img(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._process_img(obs)
        return obs


class StateLoadWrapper(gym.Wrapper):
    """
    Works with mujoco environments, useful for resetting environment in some particular state.
    """
    def __init__(self, env, obs_to_state_fn):
        super().__init__(env)
        self._obs_to_env_fn = obs_to_state_fn

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)

    def load_state(self, obs):
        state = self._obs_to_env_fn(obs)
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
        self._prev_achieved = fake_goal
        self.observation_space.spaces['prev_achieved_goal'] = copy(
            self.observation_space['achieved_goal']
        )

    def step(self, action, **kwargs):
        observation, reward, done, info = self.env.step(action, **kwargs)
        observation['prev_achieved_goal'] = self._prev_achieved
        self._prev_achieved = observation['achieved_goal']
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation['prev_achieved_goal'] = self._prev_achieved
        self._prev_achieved = observation['achieved_goal']
        return observation
