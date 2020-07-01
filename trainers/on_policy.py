import numpy as np
import torch
from tqdm import trange
from tensorboardX import SummaryWriter


from algorithms.normalization import RunningMeanStd


# TODO: add logging here
class OnPolicyTrainer:
    """
    Simple on-policy trainer.
    """
    def __init__(
            self,
            agent, train_env, test_env,
            normalize_obs, normalize_reward,
            reward_clip_min, reward_clip_max,
            log_dir
    ):
        self._agent = agent
        # both environments should:
        #   vectorized
        #   reset environment automatically
        self._train_env = train_env
        self._test_env = test_env

        # normalizers:
        self._obs_normalizer = RunningMeanStd() if normalize_obs else None
        self._reward_normalizer = RunningMeanStd() if normalize_reward else None
        self._reward_clip_min = reward_clip_min
        self._reward_clip_max = reward_clip_max

        # store episode reward and number for each train environment
        self._env_reward = np.zeros(train_env.num_envs, dtype=np.float32)
        self._env_episode = np.zeros(train_env.num_envs, dtype=np.int32)

        self._log_dir = log_dir

        self._writer = SummaryWriter(log_dir + 'tb/')

    def save(self, filename):
        state_dict = {'agent': self._agent.state_dict()}
        if self._obs_normalizer is not None:
            state_dict['obs_normalizer'] = self._obs_normalizer.state_dict()
        if self._reward_normalizer is not None:
            state_dict['reward_normalizer'] = self._reward_normalizer.state_dict()
        torch.save(state_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self._agent.load_state_dict(checkpoint['agent'])
        if 'obs_normalizer' in checkpoint:
            self._obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
        if 'reward_normalizer' in checkpoint:
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

    def _act(self, observation, deterministic):
        with torch.no_grad():
            action = self._agent.act(observation, deterministic)
        return action.cpu().numpy()

    def _gather_rollout(self, observation, rollout_len):
        # this function only called when agent is trained
        # initial observation must be normalized
        observations, actions, rewards, is_done = [observation], [], [], []
        rollout_mean_reward = 0.0  # raw reward from environment
        for _ in range(rollout_len):
            # on-policy trainer does not requires actions to be differentiable
            # however, agent may be used by different algorithms which may require that
            with torch.no_grad():
                action = self._act(observation, deterministic=False)

            observation, reward, done, _ = self._train_env.step(action)
            self._env_reward += reward  # add raw reward here
            reward = np.clip(reward, self._reward_clip_min, self._reward_clip_max)

            rollout_mean_reward += np.mean(reward)

            if self._obs_normalizer is not None:
                self._obs_normalizer.update(observation)
                mean, var = self._obs_normalizer.mean, self._obs_normalizer.var
                observation = (observation - mean) / np.sqrt(var + 1e-8)

            if self._reward_normalizer is not None:
                self._reward_normalizer.update(reward)
                mean, var = self._reward_normalizer.mean, self._reward_normalizer.var
                reward = (reward - mean) / np.sqrt(var + 1e-8)

            observations.append(observation)
            actions.append(action)
            is_done.append(done)

            rewards.append(reward)
            self._done_callback(done)

        rollout = observations, actions, rewards, is_done
        rollout_mean_reward /= rollout_len
        return rollout, observation, rollout_mean_reward

    def _done_callback(self, done):
        if np.any(done):
            for i, d in enumerate(done):
                if d:
                    self._write_logs(
                        f'agents/agent_{i}/',
                        {'reward': self._env_reward[i]},
                        self._env_episode[i]
                    )
                    self._env_reward[i] = 0
                    self._env_episode[i] += 1

    def _write_logs(self, tag, values, step):
        for key, value in values.items():
            self._writer.add_scalar(tag + key, value, step)

    def _train_step(self, observation, rollout_len, step):
        # gather rollout -> train on it -> write training logs
        rollout, observation, rollout_mean_reward = self._gather_rollout(observation, rollout_len)
        train_logs = self._agent.train_on_rollout(rollout)
        train_logs['reward'] = rollout_mean_reward
        self._write_logs('train/', train_logs, step)
        return observation

    def _test_agent(self, step, n_tests):
        self._agent.eval()
        n_test_envs = self._test_env.num_envs
        observation = self._test_env.reset()
        env_reward = np.zeros(n_test_envs, dtype=np.float32)
        episode_rewards = []

        while len(episode_rewards) < n_tests:
            if self._obs_normalizer is not None:
                mean, var = self._obs_normalizer.mean, self._obs_normalizer.var
                observation = (observation - mean) / np.sqrt(var + 1e-8)
            action = self._act(observation, deterministic=True)
            observation, reward, done, _ = self._test_env.step(action)
            env_reward += reward
            if np.any(done):
                for i, d in enumerate(done):
                    if d:
                        episode_rewards.append(env_reward[i])
                        env_reward[i] = 0.0

        reward_mean = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        write_dict = {
            'reward_mean': reward_mean,
            'reward_std': reward_std
        }
        self._write_logs('test/', write_dict, step)

    def train(self, n_epoch, n_steps, rollout_len, n_tests):
        """
        Run training for 'n_epoch', each epoch takes 'n_steps' training steps
        on rollouts of len 'rollout_len'.
        At the end of each epoch run 'n_tests' tests and saves checkpoint

        :param n_epoch:
        :param n_steps:
        :param rollout_len:
        :param n_tests:
        :return:
        """
        observation = self._train_env.reset()
        if self._obs_normalizer is not None:
            self._obs_normalizer.update(observation)
            mean, var = self._obs_normalizer.mean, self._obs_normalizer.var
            observation = (observation - mean) / np.sqrt(var + 1e-8)
        self._test_agent(0, n_tests)

        for epoch in range(n_epoch):
            self._agent.train()
            p_bar = trange(n_steps, ncols=90, desc=f'epoch_{epoch}')
            for train_step in p_bar:
                observation = self._train_step(
                    observation, rollout_len, train_step + epoch * n_steps
                )
            self._agent.eval()
            self._test_agent(epoch + 1, n_tests)
            checkpoint_name = self._log_dir + 'checkpoints/' + f'epoch_{epoch}.pth'
            self.save(checkpoint_name)
        self._writer.close()
