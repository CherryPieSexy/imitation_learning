import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.actor_critic import ActorCriticTwoMLP
from algorithms.agents.base_agent import AgentInference
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


train_env_num = 4
test_env_num = 4
distribution = 'Categorical'
device = torch.device('cpu')
log_dir = 'logs_py/cart_pole/exp_1/'

actor_critic_args = {
    'observation_size': 5, 'hidden_size': 16, 'action_size': 2,
    'distribution': distribution
}
agent_train_args = {
    'learning_rate': 0.001,
    'ppo_n_epoch': 4, 'ppo_n_mini_batches': 4,
    'rollback_alpha': 0.1
}
trainer_args = {'n_plot_agents': 1, 'log_dir': log_dir}
train_args = {
    'n_epoch': 10, 'n_steps_per_epoch': 100,
    'rollout_len': 16, 'n_tests_per_epoch': 100,
}


def make_env():
    def make():
        env = gym.make('CartPole-v1')
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        env = wrappers.TimeStepWrapper(env, 500)
        return env
    return make


def make_vec_env(n_envs):
    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    return vec_env


def make_agent_online():
    actor_critic_online = ActorCriticTwoMLP(**actor_critic_args)
    agent = AgentInference(
        actor_critic_online, device, distribution
    )
    return agent


def make_agent_train():
    actor_critic_train = ActorCriticTwoMLP(**actor_critic_args)
    agent = PPO(
        actor_critic_train, device, distribution,
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
