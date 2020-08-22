import numpy as np
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter


from utils.utils import time_it


class BehaviorCloningTrainer:
    def __init__(
            self,
            agent, test_env,
            demo_buffer, log_dir
    ):
        self._agent = agent
        self._test_env = test_env
        self._demo_buffer = demo_buffer

        self._log_dir = log_dir
        self._writer = SummaryWriter(log_dir + 'tb_logs/')

    def save(self, filename):
        state_dict = {'agent': self._agent.state_dict()}
        torch.save(state_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self._agent.load_state_dict(checkpoint['agent'])

    @time_it
    def _act(self, observation, deterministic):
        # add and remove time dimension into observation
        with torch.no_grad():
            action, _ = self._agent.act([observation], False, deterministic)
        return action.cpu().numpy()[0]

    @staticmethod
    @time_it
    def env_step(env, action):
        return env.step(action)

    def _write_logs(self, tag, values, step):
        for key, value in values.items():
            self._writer.add_scalar(tag + key, value, step)

    @time_it
    def _test_agent_service(self, n_tests):
        # do the job
        n_test_envs = self._test_env.num_envs
        observation = self._test_env.reset()
        env_reward = np.zeros(n_test_envs, dtype=np.float32)
        episode_rewards = []

        while len(episode_rewards) < n_tests:
            # I do not care about time while testing
            action, _ = self._act(observation, deterministic=True)
            env_step_result, _ = self.env_step(self._test_env, action)
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

    def _test_agent(self, step, n_tests):
        # call the testing function and write logs
        # self._agent.eval()
        test_result, test_time = self._test_agent_service(n_tests)
        # self._agent.train()
        test_result['test_time'] = test_time
        self._write_logs('test/', test_result, step)

    def _train_step(self, batch, step):
        train_logs, time_logs, = self._agent.train_on_rollout(batch)
        self._write_logs('train/', train_logs, step)
        self._write_logs('time/', time_logs, step)

    def train(self, n_epoch, n_tests):
        # self._test_agent(0, n_tests)
        self._agent.train()
        train_step = 0

        for epoch in range(n_epoch):
            for batch in tqdm(self._demo_buffer, ncols=90, desc=f'epoch_{epoch}'):
                self._train_step(batch, train_step)
                train_step += 1
            checkpoint_name = self._log_dir + 'checkpoints/' + f'epoch_{epoch}.pth'
            self.save(checkpoint_name)
            self._test_agent(epoch + 1, n_tests)
        self._writer.close()
