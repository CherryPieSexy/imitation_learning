from time import sleep

import gym
import vizdoom
import numpy as np


ratios = {
    '4:3': vizdoom.ScreenResolution.RES_640X480,
    '16:9': vizdoom.ScreenResolution.RES_640X360,
    '16:10': vizdoom.ScreenResolution.RES_640X400
}

TURN_LEFT = vizdoom.vizdoom.Button.TURN_LEFT
TURN_RIGHT = vizdoom.vizdoom.Button.TURN_RIGHT


class Doom:
    """
    VizDoom environment with gym interface.
    """
    def __init__(
            self,
            scenario,
            aspect_ratio=None,
            set_window_visible=False,
            action_repeat=4
    ):
        """
        :param scenario: scenario name w/o '.cfg' at the end.
        :param aspect_ratio: width to high ratio, one from ['4:3', '16:9', '16:10'].
        :param set_window_visible: bool.
        :param action_repeat: number of consecutive env steps with given action.
        """
        env = vizdoom.DoomGame()
        try:
            env.load_config(vizdoom.scenarios_path + '/' + scenario + '.cfg')
        except vizdoom.vizdoom.FileDoesNotExistException:
            env.load_config('configs/doom/scenarios/' + scenario + '.cfg')

        aspect_ratio = '16:9' if aspect_ratio is None else aspect_ratio
        env.set_screen_resolution(ratios[aspect_ratio])
        env.set_window_visible(set_window_visible)
        self._env = env
        self._action_repeat = action_repeat

        n_buttons = self._env.get_available_buttons_size()
        self.action_space = gym.spaces.MultiBinary(n_buttons)

        self._env.init()

        self._current_img = None
        self._current_info = None
        self._info_keys = self._env.get_available_game_variables()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, dtype=np.uint8,
            shape=(
                self._env.get_screen_height(),
                self._env.get_screen_width(),
                self._env.get_screen_channels()
            )
        )

        self.reward_range = (-float('inf'), float('inf'))
        self.spec = None
        self.metadata = None

    def _get_img_and_info(self):
        # there is also [depth_buffer, labels_buffer, automap_buffer, labels] presented in state
        state = self._env.get_state()
        img = state.screen_buffer
        img = np.transpose(img, (1, 2, 0))  # (c, h, w) -> (h, w, c)

        info = {k: v for k, v in zip(self._info_keys, state.game_variables)}

        self._current_img = img
        self._current_info = info

    # noinspection PyUnboundLocalVariable
    def step(self, action, render=False):
        step_reward = 0.0
        action_list = action.tolist()

        for _ in range(self._action_repeat):
            if render:
                sleep(1 / 60)

            step_reward += self._env.make_action(action_list)
            done = self._env.is_episode_finished()
            if done:
                break

        if not done:
            self._get_img_and_info()

        return self._current_img, step_reward, done, self._current_info

    def reset(self):
        self._env.new_episode()
        self._get_img_and_info()
        return self._current_img

    # noinspection PyUnusedLocal
    def render(self, *args, **kwargs):
        self._env.set_window_visible(True)
        sleep(1 / 60)
        return self._current_img

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._env.set_seed(seed)
        return


def main():
    from cherry_rl.utils.image_wrapper import ImageEnvWrapper
    environment = Doom('defend_the_line_walk', set_window_visible=True)
    environment = ImageEnvWrapper(environment, x_size=128, y_size=72)

    for _ in range(5):
        environment.reset()
        #     environment.render()
        env_done = False
        while not env_done:
            action = environment.action_space.sample()
            img, _, env_done, _ = environment.step(action)
            environment.render()


if __name__ == '__main__':
    main()
