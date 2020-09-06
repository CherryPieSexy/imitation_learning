import numpy as np
import torch
from tensorboardX import SummaryWriter

from utils.utils import time_it


class BaseTrainer:
    def __init__(self, test_env, log_dir):
        self._test_env = test_env

        self._log_dir = log_dir
        self._writer = SummaryWriter(log_dir + 'tb_logs/')

    @staticmethod
    @time_it
    def _act(
            agent, observation, deterministic,
            return_pi=False, require_grad=False,
            to_numpy=True
    ):
        """Method to get an action from the agent.

        :param agent: an agent to sample action, expect observations of shape [Time, Batch, dim(obs)]
        :param observation: np.array of batch of observations, shape [Batch, dim(obs)]
        :param deterministic: if True then action will be chosen as policy mean
        :param return_pi: default to False, if True then returns full policy parameters
        :param require_grad: default to False, if True then action will have gradients
        :param to_numpy: defaults to True, if False then return
                         action and log-prob as 'torch.tensor' instances
        :return: action and log-prob of action or full policy parameters
        """

        if require_grad:
            action, log_prob = agent.act([observation], return_pi, deterministic)
        else:
            with torch.no_grad():
                action, log_prob = agent.act([observation], return_pi, deterministic)
        action, log_prob = action[0], log_prob[0]
        if to_numpy:
            action, log_prob = action.cpu().numpy(), log_prob.cpu().numpy()
        return action, log_prob

    @staticmethod
    @time_it
    def _env_step(env, action):
        return env.step(action)

    def _write_logs(self, tag, values, step):
        for key, value in values.items():
            self._writer.add_scalar(tag + key, value, step)

    @time_it
    def _test_agent_service(self, n_tests, agent, deterministic):
        """Tests an 'agent' by playing #'n_test' episodes and return result

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
            (action, _), _ = self._act(agent, observation, deterministic=deterministic)
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
