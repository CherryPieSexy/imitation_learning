import time

import torch
import numpy as np
from tqdm import trange

from utils.utils import time_it
from trainers.base_trainer import BaseTrainer
from trainers.rollout import Rollout


# TODO: add logging here
class OnPolicyTrainer(BaseTrainer):
    """
    On-policy trainer.

    Controls data gathering from environment by online agent,
    logs environment statistics (reward),
    trains agent on gathered data,
    periodically (once per epoch) tests and saves agent.
    """
    def __init__(
            self,
            agent, train_env,
            recurrent,
            warm_up_steps=0,
            n_plot_agents=1,
            average_n_episodes=20,
            **kwargs
    ):
        """
        :param agent: agent for gathering data and train on it.
        :param train_env: environment for collecting training data.
        :param warm_up_steps: number of steps not to update agent,
                              useful to getting accurate running statistics.
        :param plot_agents: number of agent statistics to be plotted in tensorboard.
        :param average_n_returns: number of agent returns to be averaged
                                  and plotted in tensorboard.
        :param kwargs: test_env and log_dir
        """
        super().__init__(**kwargs)

        self._agent = agent
        self._agent_memory = None
        self._train_env = train_env
        self._recurrent = recurrent

        self._warm_up_steps = warm_up_steps

        self._n_plot_agents = n_plot_agents
        self._average_n_episodes = average_n_episodes
        self._last_total_rewards = []
        self._last_lengths = []

        self._gamma = self._agent.gamma
        # store episode reward, length, return and number for each train environment
        self._alive_env = np.array([False for _ in range(self._train_env.num_envs)])
        self._env_total_reward = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_discounted_return = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode_len = np.zeros(train_env.num_envs, dtype=np.int32)
        self._env_episode_number = np.zeros(train_env.num_envs, dtype=np.int32)

    def _update_running_statistics(self, raw_reward):
        self._env_total_reward += raw_reward * self._alive_env
        self._env_episode_len += 1 * self._alive_env
        self._env_discounted_return = self._gamma * self._env_discounted_return + raw_reward * self._alive_env

    def _plot_agent(self, i):
        if self._writer is not None:
            self._writer.add_scalars(
                'agents/train_reward/',
                {f'agent_{i}': self._env_total_reward[i]},
                self._env_episode_number[i]
            )
            self._writer.add_scalars(
                'agents/train_ep_len/',
                {f'agent_{i}': self._env_episode_len[i]},
                self._env_episode_number[i]
            )

    def _plot_average(self):
        if self._writer is not None:
            self._writer.add_scalars(
                'agents/train_reward/',
                {f'agent_mean': sum(self._last_total_rewards) / self._average_n_episodes},
                self._env_episode_number[0]
            )
            self._writer.add_scalars(
                'agents/train_ep_len/',
                {f'agent_mean': sum(self._last_lengths) / self._average_n_episodes},
                self._env_episode_number[0]
            )
        self._last_total_rewards = []
        self._last_lengths = []

    def _small_done_callback(self, i):
        self._last_total_rewards.append(self._env_total_reward[i])
        self._last_lengths.append(self._env_episode_len[i])
        if len(self._last_total_rewards) == self._average_n_episodes:
            self._plot_average()

        if i < self._n_plot_agents:
            self._plot_agent(i)

        self._env_total_reward[i] = 0.0
        self._env_episode_len[i] = 0
        self._env_discounted_return[i] = 0.0
        self._env_episode_number[i] += 1

    def _reset_env_and_memory_by_ids(self, observation, reset_ids):
        ids = [i for i, d in enumerate(reset_ids) if d]
        self._agent_memory = self._agent.model.reset_memory_by_ids(self._agent_memory, reset_ids)
        observation[ids] = self._train_env.reset_ids(reset_ids)
        for i, idx in enumerate(reset_ids):
            if idx:
                self._alive_env[i] = True
        return observation

    def _done_callback(self, observation, done):
        if np.any(done):
            for i, d in enumerate(done):
                if d and self._alive_env[i]:
                    self._alive_env[i] = False
                    self._small_done_callback(i)
            if not self._recurrent:
                observation = self._reset_env_and_memory_by_ids(observation, done)
        return observation

    def _act_and_step(self, observation):
        act_result, act_time = time_it(self._agent.model.act)(
            observation, self._agent_memory, self._agent.device, deterministic=False
        )
        action = act_result['action']
        self._agent_memory = act_result.pop('memory')

        env_result, env_time = time_it(self._env_step)(self._train_env, action)
        observation, reward, done, info = env_result
        self._update_running_statistics(reward)
        observation = self._done_callback(observation, done)

        result = {
            'observation': observation,
            'reward': reward,
            'return': self._env_discounted_return,
            'done': done,
            'info': info,
            **act_result
        }
        return result, act_time, env_time

    def _gather_rollout(self, observation, rollout_len):
        start_time = time.time()
        rollout = Rollout(observation, self._agent_memory, self._recurrent)

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

    def _train_step(self, observation, rollout_len, step, train_agent=True):
        # gather rollout -> train on it -> write training logs
        observation, rollout, time_log = \
            self._gather_rollout(observation, rollout_len)
        rewards = rollout.get('rewards')
        mask = rollout.mask[..., None]
        mean = (mask * rewards).sum() / mask.sum()
        mean_2 = (mask * rewards ** 2).sum() / mask.sum()
        std = np.sqrt(mean_2 - mean ** 2)

        if self._recurrent:
            dead = 1.0 - self._alive_env
            if np.any(dead):
                observation = self._reset_env_and_memory_by_ids(observation, dead)

        train_logs, time_logs = self._agent.train_on_rollout(
            rollout, do_train=train_agent
        )

        train_logs.update({'reward_mean': mean, 'reward_std': std})
        time_logs.update(time_log)
        self._write_logs('train/', train_logs, step)
        self._write_logs('time/', time_logs, step)
        return observation

    def train(self, n_epoch, n_steps_per_epoch, rollout_len, n_tests_per_epoch):
        """
        Run training for 'n_epoch', each epoch takes 'n_steps' training steps
        on rollouts of len 'rollout_len'.
        At the end of each epoch run 'n_tests' trainer tests agent and saves checkpoint.

        :param n_epoch:
        :param n_steps_per_epoch:
        :param rollout_len:
        :param n_tests_per_epoch:
        :return:
        """
        observation = self._train_env.reset()
        self._alive_env = np.array([True for _ in range(self._train_env.num_envs)])
        torch.save(
            self._agent.model.state_dict(),
            self._log_dir + 'checkpoints/epoch_0.pth'
        )
        self._test_agent(0, n_tests_per_epoch, self._agent)

        if self._warm_up_steps > 0:
            # just update normalizers statistics without training agent
            p_bar = trange(self._warm_up_steps, ncols=90, desc='warm_up')
            for step in p_bar:
                observation = self._train_step(
                    observation, rollout_len, step, train_agent=False
                )

        for epoch in range(n_epoch):
            self._agent.model.train()
            p_bar = trange(n_steps_per_epoch, ncols=90, desc=f'epoch_{epoch}')
            for train_step in p_bar:
                step = train_step + epoch * n_steps_per_epoch + self._warm_up_steps
                observation = self._train_step(
                    observation, rollout_len, step
                )
            torch.save(
                self._agent.model.state_dict(),
                self._log_dir + 'checkpoints/' + f'epoch_{epoch + 1}.pth'
            )
            self._test_agent(epoch + 1, n_tests_per_epoch, self._agent)
        if self._writer is not None:
            self._writer.close()
