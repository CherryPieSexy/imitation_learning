import cv2
import numpy as np
from gym import Wrapper
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage

    def step(self, action):
        state, reward, done, info = self.env.step(int(action))
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                penalty = 500 if self.world == 2 and self.stage == 2 else 50
                reward -= penalty
        if self.world in [1, 4] and self.stage == 2:
            # prevents Mario from running into "warp zone"
            if info["y_pos"] >= 255:
                reward -= 50
                done = True
        if self.world == 1 and self.stage == 3:
            if (640 <= info["x_pos"] <= 660 and info["y_pos"] < 190) or (
                    1661 <= info["x_pos"] <= 1680 and info["y_pos"] < 131):
                reward -= 50
                done = True
        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action, render=False):
        total_reward = 0
        last_states = []
        done, info = False, dict()
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            if render:
                self.env.render()
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states.astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states.astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states.astype(np.float32)


def make_mario_env(world, stage, complex_movement=False):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))

    env = JoypadSpace(env, COMPLEX_MOVEMENT if complex_movement else SIMPLE_MOVEMENT)
    env = CustomReward(env, world, stage)
    env = CustomSkipFrame(env)
    return env
