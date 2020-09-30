import numpy as np
import torch
from tqdm import trange


from utils.utils import time_it
from algorithms.normalization import RunningMeanStd
from trainers.base_trainer import BaseTrainer


# TODO: add logging here
class OnPolicyTrainer(BaseTrainer):
    def __init__(
            self,
            agent_online, agent_train,
            return_pi,
            train_env,
            update_period,
            normalize_obs, train_obs_normalizer,
            scale_reward, normalize_reward, train_reward_normalizer,
            obs_clip, reward_clip,
            warm_up_steps=0,
            **kwargs
    ):
        """On-policy trainer

        :param agent_online: agent which collects data
        :param agent_train: agent which performs train-ops
        :param return_pi: if True then 'act()' method will return
                          full policy parametrization instead of log-prob of sampled action
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

        # if return_pi then agent append 'policy' tensor
        # to rollout, else it append 'log_pi_a'
        self._return_pi = return_pi
        # both environments should:
        #   vectorized
        #   reset environment automatically
        self._train_env = train_env

        # normalizers:
        self._obs_normalizer = RunningMeanStd() if normalize_obs else None
        self._train_obs_normalizer = train_obs_normalizer
        self._obs_clip = obs_clip
        assert not (normalize_reward and scale_reward), \
            'reward may be normalized or scaled, but not both at the same time!'
        self._reward_scaler = RunningMeanStd() if scale_reward else None
        self._reward_normalizer = RunningMeanStd() if normalize_reward else None
        self._train_reward_normalizer = train_reward_normalizer
        self._reward_clip = reward_clip

        # store episode reward, length, return and number for each train environment
        self._gamma = self._agent_train.gamma
        self._env_reward = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode_len = np.zeros(train_env.num_envs, dtype=np.int32)
        self._env_return = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode = np.zeros(train_env.num_envs, dtype=np.int32)

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

    @staticmethod
    def stack_infos(infos):
        keys = infos[0].keys()
        stacked_info = dict()
        for key in keys:
            values = []
            for info in infos:
                values.append(info[key])
            stacked_info[key] = np.stack(values)
        return stacked_info

    def _act(self, fake_agent, observation, deterministic, need_norm=True, **kwargs):
        # this method used ONLY inside base class '._test_agent_service()' method
        # 'fake_agent' arg is unused to make this method work in BaseTrainer.test_agent_service
        if need_norm:
            observation = self._normalize_observation(observation, False)
        (action, log_prob), act_time = super()._act(
            self._agent_online, observation, deterministic,
            return_pi=self._return_pi
        )
        return (action, log_prob), act_time

    def _normalize_observation(self, observation, training):
        if self._obs_normalizer is not None:
            if training:
                self._obs_normalizer.update(observation)
            mean, var = self._obs_normalizer.mean, self._obs_normalizer.var
            observation = (observation - mean) / np.sqrt(var + 1e-8)
            observation = np.clip(observation, -self._obs_clip, self._obs_clip)
        return observation

    def _normalize_reward(self, reward, training):
        # 'baselines' version:
        if self._reward_scaler is not None:
            if training:
                self._reward_scaler.update(self._env_return)
            var = self._reward_scaler.var
            reward = reward / np.sqrt(var + 1e-8)
            reward = np.clip(reward, -self._reward_clip, self._reward_clip)

        # 'my' version:
        if self._reward_normalizer is not None:
            if training:
                self._reward_normalizer.update(reward)
            mean, var = self._reward_normalizer.mean, self._reward_normalizer.var
            reward = (reward - mean) / np.sqrt(var + 1e-8)
            reward = np.clip(reward, -self._reward_clip, self._reward_clip)

        return reward

    @time_it
    def _gather_rollout(self, observation, rollout_len):
        # this function only called when agent is trained
        # initial observation (i.e. at the beginning of training) does not care about normalization
        observations, actions, rewards, is_done = [observation], [], [], []
        log_probs = []
        raw_rewards = []

        mean_act_time = 0
        mean_env_time = 0

        for _ in range(rollout_len):
            # on-policy trainer does not requires actions to be differentiable
            # however, agent may be used by different algorithms which may require that
            (action, log_prob), act_time = self._act(None, observation, deterministic=False, need_norm=False)
            mean_act_time += act_time

            env_step_result, env_time = self._env_step(self._train_env, action)
            observation, reward, done, _ = env_step_result
            mean_env_time += env_time

            self._env_reward += reward
            self._env_episode_len += 1
            self._env_return = reward + self._gamma * self._env_return

            raw_rewards.append(np.copy(reward))

            observation = self._normalize_observation(observation, training=self._train_obs_normalizer)
            reward = self._normalize_reward(reward, training=self._train_reward_normalizer)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            is_done.append(done)
            log_probs.append(log_prob)

            self._done_callback(done)

        mean_act_time /= rollout_len
        mean_env_time /= rollout_len

        rollout = observations, actions, rewards, is_done, log_probs
        gather_result = rollout, raw_rewards, observation
        mean_time = mean_act_time, mean_env_time
        return gather_result, mean_time

    def _done_callback(self, done):
        if np.any(done):
            for i, d in enumerate(done):
                if d:
                    self._writer.add_scalars(
                        'agents/train_reward/',
                        {f'agent_{i}': self._env_reward[i]},
                        self._env_episode[i]
                    )
                    self._writer.add_scalars(
                        'agents/train_ep_len/',
                        {f'agent_{i}': self._env_episode_len[i]},
                        self._env_episode[i]
                    )
                    self._env_reward[i] = 0.0
                    self._env_episode_len[i] = 0
                    self._env_return[i] = 0.0
                    self._env_episode[i] += 1

    def _train_step(self, observation, rollout_len, step):
        # gather rollout -> train on it -> write training logs
        (gather_result, mean_time), gather_time = self._gather_rollout(observation, rollout_len)

        rollout, raw_rewards, observation = gather_result
        mean_act_time, mean_env_time = mean_time

        train_logs, time_logs = self._agent_train.train_on_rollout(rollout)
        train_logs['reward_mean'] = np.mean(raw_rewards)
        train_logs['reward_std'] = np.std(raw_rewards)

        time_logs['mean_act_time'] = mean_act_time
        time_logs['mean_env_time'] = mean_env_time
        time_logs['gather_rollout_time'] = gather_time

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
        for epoch in range(n_epoch):
            self._agent_online.train()
            p_bar = trange(n_steps_per_epoch, ncols=90, desc=f'epoch_{epoch}')
            for train_step in p_bar:
                step = train_step + epoch * n_steps_per_epoch
                observation = self._train_step(
                    observation, rollout_len, step
                )
                if step > self._warm_up_steps and (step + 1) % self._update_period == 0:
                    self._update_online_agent()

            self._update_online_agent()
            self._save_n_test(epoch + 1, n_tests_per_epoch, self._agent_online)
        self._writer.close()
