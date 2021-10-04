import time

import torch
import numpy as np
from tqdm import trange

from cherry_rl.utils.utils import time_it
from cherry_rl.utils.vec_env import SubprocVecEnv
from cherry_rl.algorithms.rollout import Rollout


# noinspection PyUnusedLocal
def default_select_reward_to_plot(reward, idx):
    """
    Function to select reward for plotting in case of vectorized reward.
    Useful for two-players games.

    :param reward: tensor or numpy array of shape [reward size],
    :param idx:    index of environment from reward was obtained.
    :return:       scalar reward
    """
    return reward


class TrainAgent:
    """
    Basically copy of the old 'OnPolicyTrainer' but with queues and pipes.
    """
    def __init__(
            self,
            make_env, train_env_num,
            gamma, recurrent, log_dir,
            queue_to_model,  # send observation to model
            queue_to_optimizer,  # send rollout to train algo
            pipe_from_model,  # receive actions from online model
            queue_to_tb_writer,  # send logs to writing
            n_plot_agents=1,
            average_n_episodes=20,
            reward_size=1,
            select_reward_to_plot_fn=default_select_reward_to_plot
    ):
        self._env = SubprocVecEnv([make_env() for _ in range(train_env_num)])
        self._gamma = gamma
        self._model_memory = None
        self._recurrent = recurrent
        self._log_dir = log_dir

        self._queue_to_model = queue_to_model
        self._queue_to_optimizer = queue_to_optimizer
        self._pipe_from_model = pipe_from_model
        self._queue_to_tb_writer = queue_to_tb_writer

        self._n_plot_agents = n_plot_agents
        self._average_n_episodes = average_n_episodes

        self._env_total_reward = np.zeros((self._env.num_envs, reward_size), dtype=np.float32)
        self._env_discounted_return = np.zeros((self._env.num_envs, reward_size), dtype=np.float32)
        self._select_reward_to_plot_fn = select_reward_to_plot_fn

        self._env_episode_len = np.zeros(self._env.num_envs, dtype=np.int32)
        self._env_episode_number = np.zeros(self._env.num_envs, dtype=np.int32)
        self._last_total_rewards = []
        self._last_lengths = []

        self._alive_env = np.array([[1] for _ in range(self._env.num_envs)])
        self._gather_steps_done = 0

    def _update_running_statistics(self, raw_reward):
        self._env_total_reward += raw_reward * self._alive_env
        self._env_episode_len += 1 * self._alive_env.squeeze(-1)
        self._env_discounted_return = self._gamma * self._env_discounted_return + raw_reward * self._alive_env

    def _plot_average(self):
        self._queue_to_tb_writer.put((
            'add_scalars',
            (
                'agents/train_reward/',
                {'agent_mean': sum(self._last_total_rewards) / self._average_n_episodes},
                self._env_episode_number[0]
            )
        ))
        self._queue_to_tb_writer.put((
            'add_scalars',
            (
                'agents/train_ep_len/',
                {'agent_mean': sum(self._last_lengths) / self._average_n_episodes},
                self._env_episode_number[0]
            )
        ))
        self._last_total_rewards = []
        self._last_lengths = []

    def _plot_agent(self, i):
        self._queue_to_tb_writer.put((
            'add_scalars',
            (
                'agents/train_reward/',
                {f'agent_{i}': self._select_reward_to_plot_fn(self._env_total_reward[i], i)},
                self._env_episode_number[i]
            )
        ))
        self._queue_to_tb_writer.put((
            'add_scalars',
            (
                'agents/train_ep_len/',
                {f'agent_{i}': self._env_episode_len[i]},
                self._env_episode_number[i]
            )
        ))

    def _small_done_callback(self, i):
        selected_reward = self._select_reward_to_plot_fn(self._env_total_reward[i], i)
        self._last_total_rewards.append(np.copy(selected_reward))
        self._last_lengths.append(self._env_episode_len[i])
        if len(self._last_total_rewards) == self._average_n_episodes:
            self._plot_average()

        if i < self._n_plot_agents:
            self._plot_agent(i)

        self._env_total_reward[i] *= 0
        self._env_episode_len[i] = 0
        self._env_episode_number[i] += 1

    def _reset_discounted_returns(self, done):
        for i, d in enumerate(done):
            # in recurrent case it will be probably masked anyway.
            if d:
                self._env_discounted_return[i] = 0.0

    def _reset_env_and_memory_by_ids(self, observation, reset_ids):
        ids = [i for i, d in enumerate(reset_ids) if d]
        self._queue_to_model.put((
            'reset_memory', 'train_agent', (self._model_memory, reset_ids)
        ))
        model_memory = self._pipe_from_model.recv()
        self._model_memory = self._clone(model_memory)
        del model_memory
        reset_observations = self._env.reset_ids(reset_ids)
        # TODO: this MUST be carefully tested.
        if type(observation) is dict:
            for k, v in observation.items():
                v[ids] = reset_observations[k]
        else:
            observation[ids] = reset_observations
        for i, idx in enumerate(reset_ids):
            if idx:
                self._alive_env[i, 0] = True
        return observation

    def _done_callback(self, observation, done):
        if np.any(done):
            for i, d in enumerate(done):
                if d and self._alive_env[i, 0]:
                    self._alive_env[i, 0] = 0
                    self._small_done_callback(i)
            if not self._recurrent:
                observation = self._reset_env_and_memory_by_ids(observation, done)
        return observation

    @time_it
    def _env_step(self, action):
        if type(action) is tuple:
            env_action = tuple([
                [mode[i].cpu().numpy() for mode in action]
                for i in range(len(action[0]))
            ])
        else:
            env_action = action.cpu().numpy()

        return self._env.step(env_action)

    def _clone(self, x):
        if x is None:
            return None
        elif type(x) is tuple:
            return tuple([self._clone(xx) for xx in x])
        else:
            return x.clone()

    def _act_and_step(self, observation):
        start_time = time.time()
        self._queue_to_model.put((
            'act', 'train_agent', (observation, self._model_memory, False)
        ))
        act_result = self._pipe_from_model.recv()
        act_result_clone = {
            k: self._clone(v)
            for k, v in act_result.items()
        }
        del act_result
        act_time = time.time() - start_time
        action = act_result_clone['action']
        self._model_memory = act_result_clone.pop('memory')

        env_result, env_time = self._env_step(action)
        observation, reward, done, info = env_result
        if len(reward.shape) == 1:
            reward = reward[..., None]
        self._update_running_statistics(reward)
        observation = self._done_callback(observation, done)

        result = {
            'observation': observation,
            'reward': reward,
            'return': self._env_discounted_return,
            'done': done,
            'info': info,
            **act_result_clone
        }
        self._reset_discounted_returns(done)
        return result, act_time, env_time

    def _gather_rollout(self, observation, rollout_len):
        start_time = time.time()
        rollout = Rollout(observation, self._model_memory, self._recurrent)

        mean_act_time = 0
        mean_env_time = 0

        for _ in range(rollout_len):
            act_step_result, act_time, env_time = self._act_and_step(observation)
            observation = act_step_result['observation']
            rollout.append(act_step_result)
            mean_act_time += act_time
            mean_env_time += env_time

        mean_act_time /= rollout_len
        mean_env_time /= rollout_len

        elapsed_time = time.time() - start_time

        time_log = {
            'mean_act_time': mean_act_time,
            'mean_env_time': mean_env_time,
            'gather_rollout_time': elapsed_time
        }
        return observation, rollout, time_log

    def _reward_statistics(self, rollout):
        environment_indices = [[[i] for i in range(self._env.num_envs)]]  # add time and reward dimensions
        rewards = self._select_reward_to_plot_fn(
            rollout.get('rewards'),
            torch.tensor(environment_indices)
        )
        mask = rollout.get('mask').unsqueeze(-1)
        mean = (mask * rewards).sum() / mask.sum()
        mean_2 = (mask * rewards ** 2).sum() / mask.sum()
        std = np.sqrt(mean_2 - mean ** 2)
        return {'reward_mean': mean.item(), 'reward_std': std.item()}

    def _train_step(self, observation, rollout_len):
        observation, rollout, time_log = \
            self._gather_rollout(observation, rollout_len)
        self._gather_steps_done += 1

        self._queue_to_optimizer.put(('train', rollout.as_dict()))
        # cmd, (tag, logs_dict, step)
        self._queue_to_tb_writer.put((
            'add_scalar', ('time/', time_log, self._gather_steps_done)
        ))
        self._queue_to_tb_writer.put((
            'add_scalar', ('train/', self._reward_statistics(rollout), self._gather_steps_done)
        ))

        if self._recurrent:
            dead = 1.0 - self._alive_env
            if np.any(dead):
                observation = self._reset_env_and_memory_by_ids(observation, dead)

        return observation

    def train(self, n_epoch, n_steps_per_epoch, rollout_len):
        self._queue_to_optimizer.put(('save', self._log_dir + 'checkpoints/epoch_0.pth'))
        observation = self._env.reset()
        self._alive_env = np.array([[1] for _ in range(self._env.num_envs)])

        for epoch in range(n_epoch):
            p_bar = trange(n_steps_per_epoch, ncols=90, desc=f'epoch_{epoch + 1}')
            for _ in p_bar:
                observation = self._train_step(observation, rollout_len)
            self._queue_to_optimizer.put(('save', self._log_dir + f'checkpoints/epoch_{epoch + 1}.pth'))

    def close(self):
        self._env.close()
