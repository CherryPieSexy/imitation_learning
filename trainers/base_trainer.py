import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.utils import time_it


class BaseTrainer:
    def __init__(self, test_env, log_dir):
        self._test_env = test_env

        self._log_dir = log_dir
        self._writer = SummaryWriter(log_dir + 'tb_logs/')

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @time_it
    def _act(agent, observation, deterministic):
        # I do need additional method for acting to easily inherit from it
        return agent.act(observation, deterministic)

    @staticmethod
    @time_it
    def _env_step(env, action):
        return env.step(action)

    def _write_logs(self, tag, values, step):
        for key, value in values.items():
            self._writer.add_scalar(tag + key, value, step)

    @time_it
    def _test_agent_service(self, n_tests, agent, deterministic):
        """Tests an 'agent' by playing 'n_test' episodes and return result

        :param n_tests:
        :param agent:
        :return: dict reward mean & std
        """
        n_test_envs = self._test_env.num_envs
        observation = self._test_env.reset()
        env_reward = np.zeros(n_test_envs, dtype=np.float32)
        episode_rewards = []

        while len(episode_rewards) < n_tests:
            # I do not care about time while testing
            act_result, _ = self._act(agent, observation, deterministic=deterministic)
            action = act_result['action']
            env_step_result, _ = self._env_step(self._test_env, action)
            observation, reward, done, _ = env_step_result
            env_reward += reward
            if np.any(done):
                for i, d in enumerate(done):
                    if d:
                        episode_rewards.append(env_reward[i])
                        env_reward[i] = 0.0

        reward_mean = np.mean(episode_rewards)
        reward_std = np.std(episode_rewards)
        test_result = {
            'reward_mean': reward_mean,
            'reward_std': reward_std
        }
        return test_result

    def _test_agent(self, step, n_tests, agent, deterministic=True):
        # call the testing function and write logs
        agent.eval()
        test_result, test_time = self._test_agent_service(n_tests, agent, deterministic)
        test_result['test_time'] = test_time
        self._write_logs('test/', test_result, step)

    def _save_n_test(self, epoch, n_tests, agent):
        checkpoint_name = self._log_dir + 'checkpoints/' + f'epoch_{epoch}.pth'
        self.save(checkpoint_name)
        self._test_agent(epoch, n_tests, agent)
