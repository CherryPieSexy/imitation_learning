import time

import numpy as np
import torch
from tqdm import trange

from algorithms.normalization import RunningMeanStd
from trainers.base_trainer import BaseTrainer


# TODO: add logging here
class OnPolicyTrainer(BaseTrainer):
    def __init__(
            self,
            agent_online, agent_train,
            train_env,
            # kwargs comes from config
            update_period=1,
            normalize_obs=False, train_obs_normalizer=False,
            scale_reward=False, normalize_reward=False,
            train_reward_normalizer=False,
            obs_clip=float('inf'), reward_clip=float('inf'),
            warm_up_steps=0,
            **kwargs
    ):
        """On-policy trainer

        :param agent_online: agent which collects data
        :param agent_train: agent which performs train-ops
        :param train_env: environment for collecting training data
        :param test_env: environment for testing agent once per epoch
        :param update_period: number of train-ops after which
                              online agent loads weights of training agent
        :param normalize_obs: if True then observations will be normalized
                              by running mean and std of collected observations
        :param train_obs_normalizer: if True then running mean and std of obs_normalizer
                                     will be updated each environment step
        :param scale_reward: if True then reward will be normalized
                             by running mean and std of collected episodes return
        :param normalize_reward: if True then reward will be normalized
                                 by running mean and std of collected rewards
        :param train_reward_normalizer: if True then running mean and std of reward_normalizer
                                        will be updated each environment step
        :param obs_clip: abs(observation) will be clipped to this value after normalization
        :param reward_clip: abs(reward) will be clipped to this value after normalization
        :param warm_up_steps: number of steps not to update online agent,
                              useful to continue training from checkpoint
        :param kwargs: test_env and log_dir
        """
        super().__init__(**kwargs)

        self._agent_online = agent_online  # gather rollouts
        self._agent_train = agent_train  # do train-ops
        self._update_online_agent()
        # weights of online agent updated once in 'update_period' & at the end of training epoch
        self._update_period = update_period
        self._warm_up_steps = warm_up_steps

        # both environments should:
        #   vectorized
        #   reset environment automatically
        self._train_env = train_env

        # normalizers:
        self._obs_normalizer = RunningMeanStd(obs_clip) if normalize_obs else None
        self._train_obs_normalizer = train_obs_normalizer
        assert not (normalize_reward and scale_reward), \
            'reward may be normalized or scaled, but not both at the same time!'
        self._normalize_reward = normalize_reward
        self._scale_reward = scale_reward
        self._reward_normalizer = RunningMeanStd(reward_clip) \
            if normalize_reward or scale_reward else None
        self._train_reward_normalizer = train_reward_normalizer

        self._gamma = self._agent_train.gamma
        # store episode reward, length, return and number for each train environment
        self._env_total_reward = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode_len = np.zeros(train_env.num_envs, dtype=np.int32)
        self._env_discounted_return = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode_number = np.zeros(train_env.num_envs, dtype=np.int32)

    def save(self, filename):
        state_dict = {'agent': self._agent_train.state_dict()}
        if self._obs_normalizer is not None:
            state_dict['obs_normalizer'] = self._obs_normalizer.state_dict()
        if self._reward_normalizer is not None:
            state_dict['reward_normalizer'] = self._reward_normalizer.state_dict()
        torch.save(state_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self._agent_online.load_state_dict(checkpoint['agent'])
        self._agent_train.load_state_dict(checkpoint['agent'])
        if 'obs_normalizer' in checkpoint and self._obs_normalizer is not None:
            self._obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
        if 'reward_normalizer' in checkpoint and self._reward_normalizer is not None:
            self._reward_normalizer.load_state_dict(checkpoint['reward_normalizer'])

    def _act(self, fake_agent, observation, deterministic, **kwargs):
        # this method used ONLY inside base class '._test_agent_service()' method
        # 'fake_agent' arg is unused to make this method work in BaseTrainer.test_agent_service
        observation = self._normalize_observation_fn(observation, False)
        return super()._act(self._agent_online, observation, deterministic)

    def _normalize_observation_fn(self, observation, training):
        if self._obs_normalizer is not None:
            if training:
                self._obs_normalizer.update(observation)
            observation = self._obs_normalizer.normalize(observation)
        return observation

    def _normalize_reward_fn(self, reward, training):
        # 'baselines' version:
        if self._scale_reward is not None:
            if training:
                self._reward_normalizer.update(self._env_discounted_return)
            reward = self._reward_normalizer.scale(reward)

        # 'my' version:
        if self._reward_normalizer is not None:
            if training:
                self._reward_normalizer.update(reward)
            reward = self._reward_normalizer.normalize(reward)

        return reward

    def _update_running_statistics(self, raw_reward):
        self._env_total_reward += raw_reward
        self._env_episode_len += 1
        self._env_discounted_return = raw_reward + self._gamma * self._env_discounted_return

    def _normalize_rollout(self, rollout):
        if self._obs_normalizer is not None:
            observations = rollout['observations']
            normalized_observations = self._obs_normalizer.normalize_vec(observations)
            rollout['observations'] = normalized_observations
            if self._train_obs_normalizer:
                self._obs_normalizer.update_vec(observations)

        if self._reward_normalizer is not None:
            rewards = rollout['rewards']
            if self._normalize_reward:
                normalized_rewards = self._reward_normalizer.normalize_vec(rewards)
                rollout['rewards'] = normalized_rewards
                if self._train_reward_normalizer:
                    self._reward_normalizer.update_vec(rewards)
            elif self._scale_reward:
                normalized_rewards = self._reward_normalizer.scale_vec(rewards)
                rollout['rewards'] = normalized_rewards
                if self._train_reward_normalizer:
                    # TODO: this update is incorrect,
                    #  need to store last returns at each step during rollout gathering
                    self._reward_normalizer.update_vec(
                        self._env_discounted_return
                    )

        return rollout

    def _act_and_step(self, observation):
        # calculate action, step environment, collect results into dict
        act_result, act_time = self._act(None, observation, deterministic=False)
        action = act_result['action']

        env_step_result, env_time = self._env_step(self._train_env, action)
        observation, reward, done, info = env_step_result
        # convert info from tuple of dicts to dict of np.arrays
        info = {
            key: np.stack([info_env.get(key, None) for info_env in info])
            for key in info[0].keys()
        }
        self._done_callback(done)

        self._update_running_statistics(reward)
        # reward, observation = self._call_after_env_step(raw_reward, observation)
        result = {
            'observation': observation,
            'reward': reward,
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

        cat_infos = self._list_of_dicts_to_dict_of_np_array(rollout, 'info')

        keys = rollout[0].keys()
        keys = [k for k in keys if k not in ['observation', 'info']]
        rollout = {
            plural.get(k, k): np.stack([x[k] for x in rollout])
            for k in keys
        }
        rollout['observations'] = cat_observations
        rollout['infos'] = cat_infos
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

    def _small_done_callback(self, i):
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
        rollout = self._normalize_rollout(rollout)

        if train_agent:
            train_logs, time_logs = self._agent_train.train_on_rollout(rollout)
        else:
            train_logs, time_logs = dict(), dict()

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
        self._save_n_test(0, n_tests_per_epoch, self._agent_online)

        self._agent_train.train()  # always in training mode

        if self._warm_up_steps > 0:
            # just update normalizers statistics without training agent
            p_bar = trange(self._warm_up_steps, ncols=90, desc='warm_up')
            for step in p_bar:
                observation = self._train_step(
                    observation, rollout_len, step, train_agent=False
                )

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
            self._save_n_test(epoch + 1, n_tests_per_epoch, self._agent_online)
        self._writer.close()
