import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.actor_critic import ActorCriticTwoMLP
from algorithms.normalization import RunningMeanStd
from algorithms.agents.base_agent import AgentInference
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


train_env_num = 16
test_env_num = 4
distribution = 'Beta'
device = torch.device('cpu')
log_dir = 'logs_py/humanoid/exp_1/'

actor_critic_args = {
    'observation_size': 376 + 1, 'hidden_size': 64, 'action_size': 17,
    'distribution': distribution
}
agent_train_args = {
    'normalize_adv': True,
    'ppo_n_epoch': 10, 'ppo_n_mini_batches': 8
}
trainer_args = {'warm_up_steps': 10, 'n_plot_agents': 1, 'log_dir': log_dir}
train_args = {
    'n_epoch': 10, 'n_steps_per_epoch': 1000,
    'rollout_len': 256, 'n_tests_per_epoch': 100
}


def make_env():
    def make():
        env = gym.make('Humanoid-v3')
        env = wrappers.ContinuousActionWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        env = wrappers.TimeStepWrapper(env, 1000)
        return env
    return make


def make_vec_env(n_envs):
    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    return vec_env


def make_agent_online():
    actor_critic_online = ActorCriticTwoMLP(**actor_critic_args)
    obs_norm = RunningMeanStd()
    rew_norm = RunningMeanStd()
    value_norm = RunningMeanStd()
    agent = AgentInference(
        actor_critic_online, device, distribution,
        obs_normalizer=obs_norm,
        reward_normalizer=rew_norm,
        value_normalizer=value_norm,
    )
    return agent


def make_agent_train():
    actor_critic_train = ActorCriticTwoMLP(**actor_critic_args)
    obs_norm = RunningMeanStd()
    rew_norm = RunningMeanStd()
    value_norm = RunningMeanStd()
    agent = PPO(
        actor_critic_train, device, distribution,
        obs_normalizer=obs_norm,
        reward_normalizer=rew_norm,
        value_normalizer=value_norm,
        **agent_train_args
    )
    return agent


def main():
    create_log_dir(log_dir, __file__)

    train_env = make_vec_env(train_env_num)
    test_env = make_vec_env(test_env_num)

    agent_online = make_agent_online()
    agent_train = make_agent_train()
    agent_online.load_state_dict(agent_train.state_dict())

    trainer = OnPolicyTrainer(
        agent_online, agent_train, train_env,
        **trainer_args,
        test_env=test_env
    )
    trainer.train(**train_args)

    train_env.close()
    test_env.close()


if __name__ == '__main__':
    main()
