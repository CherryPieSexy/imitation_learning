import gym
import cv2
import numpy as np


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
        return img

    def _process_img(self, img):
        img = img[self._x_start:self._x_end, self._y_start:self._y_end]
        if self._convert_to_gray:
            img = self._img_2_gray(img)
        img = img / 127.5 - 1
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
