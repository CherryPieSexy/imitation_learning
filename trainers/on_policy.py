import time

import numpy as np
from tqdm import trange

from trainers.base_trainer import BaseTrainer


# TODO: add logging here
class OnPolicyTrainer(BaseTrainer):
    def __init__(
            self,
            agent_online, agent_train,
            train_env,
            update_period=1,
            warm_up_steps=0,
            n_plot_agents=2,
            average_n_returns=20,
            **kwargs
    ):
        """
        On-policy trainer.
        Controls data gathering from environment by online agent,
        logs environment statistics (reward),
        trains train agent on gathered data,
        periodically (once per epoch) tests and saves train agent.

        :param agent_online: agent which collects data
        :param agent_train: agent which performs train-ops
        :param train_env: environment for collecting training data
        :param update_period: number of train-ops after which
                              online agent loads weights of training agent
        :param warm_up_steps: number of steps not to update online agent,
                              useful to continue training from checkpoint
        :param plot_agents: number of agent statistics to be plotted in tensorboard.
        :param average_n_returns: number of agent returns to be averaged
                                  and plotted in tensorboard.
        :param kwargs: test_env and log_dir
        """
        super().__init__(**kwargs)

        self._agent_online = agent_online  # gather rollouts
        self._agent_train = agent_train  # do train-ops
        self._update_online_agent()
        # weights of online agent updated once in 'update_period' & at the end of training epoch
        self._update_period = update_period
        self._warm_up_steps = warm_up_steps
        self._n_plot_agents = n_plot_agents

        self._average_n_returns = average_n_returns
        self._last_total_rewards = []
        self._last_lengths = []

        # both environments should:
        #   be vectorized
        #   reset environment automatically
        self._train_env = train_env

        self._gamma = self._agent_train.gamma
        # store episode reward, length, return and number for each train environment
        self._env_total_reward = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_discounted_return = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode_len = np.zeros(train_env.num_envs, dtype=np.int32)
        self._env_episode_number = np.zeros(train_env.num_envs, dtype=np.int32)

    def _update_running_statistics(self, raw_reward):
        self._env_total_reward += raw_reward
        self._env_episode_len += 1
        self._env_discounted_return = raw_reward + self._gamma * self._env_discounted_return

    def _act_and_step(self, observation):
        # calculate action, step environment, collect results into dict
        act_result, act_time = self._act(self._agent_online, observation, deterministic=False)
        action = act_result['action']

        env_step_result, env_time = self._env_step(self._train_env, action)
        observation, reward, done, info = env_step_result
        # convert info from tuple of dicts to dict of np.arrays
        info = {
            key: np.stack([info_env.get(key, None) for info_env in info])
            for key in info[0].keys()
        }
        self._update_running_statistics(reward)
        self._done_callback(done)

        result = {
            'observation': observation,
            'reward': reward[..., None],
            'return': self._env_discounted_return[..., None],
            'done': done,
            'info': info,
            **act_result
        }
        return result, act_time, env_time

    @staticmethod
    def _list_of_dicts_to_dict_of_np_array(list_of_dicts, super_key):
        # useful to convert observations and infos
        result = {
            key: np.stack([x[super_key][key] for x in list_of_dicts])
            for key in list_of_dicts[0][super_key].keys()
        }
        return result

    def _rollout_from_list_to_dict(self, rollout, first_observation):
        plural = {
            'observation': 'observations',
            'action': 'actions',
            'reward': 'rewards',
            'return': 'returns',
            'done': 'is_done',
        }
        # observation may be of type dict so it requires 'special' concatenation
        if type(rollout[0]['observation']) is dict:
            cat_observations = self._list_of_dicts_to_dict_of_np_array(
                [{'observation': first_observation}, *rollout], 'observation'
            )
        else:
            cat_observations = np.stack(
                [x['observation'] for x in [{'observation': first_observation}, *rollout]]
            )

        # cat_infos = self._list_of_dicts_to_dict_of_np_array(rollout, 'info')

        keys = rollout[0].keys()
        keys = [k for k in keys if k not in ['observation', 'info']]
        rollout = {
            plural.get(k, k): np.stack([x[k] for x in rollout])
            for k in keys
        }
        rollout['observations'] = cat_observations
        # rollout['infos'] = cat_infos
        return rollout

    def _gather_rollout(self, observation, rollout_len):
        # this function is called only when agent is training
        start_time = time.time()

        first_observation = observation
        rollout = []

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

        # now rollout is a list of dicts, convert it to dict of 'numpy.array'
        rollout = self._rollout_from_list_to_dict(rollout, first_observation)

        elapsed_time = time.time() - start_time

        rewards = rollout['rewards']
        rollout_log = {
            'reward_mean': rewards.mean(),
            'reward_std': rewards.std()
        }
        time_log = {
            'mean_act_time': mean_act_time,
            'mean_env_time': mean_env_time,
            'gather_rollout_time': elapsed_time
        }
        return observation, rollout, rollout_log, time_log

    def _plot_agent(self, i):
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
        self._writer.add_scalars(
            'agents/train_reward/',
            {f'agent_mean': sum(self._last_total_rewards) / self._average_n_returns},
            self._env_episode_number[0]
        )
        self._writer.add_scalars(
            'agents/train_ep_len/',
            {f'agent_mean': sum(self._last_lengths) / self._average_n_returns},
            self._env_episode_number[0]
        )
        self._last_total_rewards = []
        self._last_lengths = []

    def _small_done_callback(self, i):
        self._last_total_rewards.append(self._env_total_reward[i])
        self._last_lengths.append(self._env_episode_len[i])
        if len(self._last_total_rewards) == self._average_n_returns:
            self._plot_average()

        if i < self._n_plot_agents:
            self._plot_agent(i)

        self._env_total_reward[i] = 0.0
        self._env_episode_len[i] = 0
        self._env_discounted_return[i] = 0.0
        self._env_episode_number[i] += 1

    def _done_callback(self, done):
        if np.any(done):
            for i, d in enumerate(done):
                if d:
                    self._small_done_callback(i)

    def _train_step(self, observation, rollout_len, step, train_agent=True):
        # gather rollout -> train on it -> write training logs
        observation, rollout, rollout_log, time_log = self._gather_rollout(observation, rollout_len)

        train_logs, time_logs = self._agent_train.train_on_rollout(
            rollout, do_train=train_agent
        )

        train_logs.update(rollout_log)
        time_logs.update(time_log)
        self._write_logs('train/', train_logs, step)
        self._write_logs('time/', time_logs, step)
        return observation

    def _update_online_agent(self):
        self._agent_online.load_state_dict(self._agent_train.state_dict())

    def train(self, n_epoch, n_steps_per_epoch, rollout_len, n_tests_per_epoch):
        """
        Run training for 'n_epoch', each epoch takes 'n_steps' training steps
        on rollouts of len 'rollout_len'.
        At the end of each epoch run 'n_tests' tests and saves checkpoint

        :param n_epoch:
        :param n_steps_per_epoch:
        :param rollout_len:
        :param n_tests_per_epoch:
        :return:
        """
        observation = self._train_env.reset()
        self._agent_train.save(self._log_dir + 'checkpoints/' + f'epoch_{0}.pth')
        self._test_agent(0, n_tests_per_epoch, self._agent_online)

        self._agent_train.train()  # always in training mode

        if self._warm_up_steps > 0:
            # just update normalizers statistics without training agent
            p_bar = trange(self._warm_up_steps, ncols=90, desc='warm_up')
            for step in p_bar:
                observation = self._train_step(
                    observation, rollout_len, step, train_agent=False
                )
                self._update_online_agent()

        for epoch in range(n_epoch):
            self._agent_online.train()
            p_bar = trange(n_steps_per_epoch, ncols=90, desc=f'epoch_{epoch}')
            for train_step in p_bar:
                step = train_step + epoch * n_steps_per_epoch + self._warm_up_steps
                observation = self._train_step(
                    observation, rollout_len, step
                )
                if (step + 1) % self._update_period == 0:
                    self._update_online_agent()

            self._update_online_agent()
            self._agent_train.save(self._log_dir + 'checkpoints/' + f'epoch_{epoch + 1}.pth')
            self._test_agent(epoch + 1, n_tests_per_epoch, self._agent_online)
        self._writer.close()
