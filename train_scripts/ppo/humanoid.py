import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.agent_model import AgentModel
from algorithms.nn.actor_critic import ActorCriticTwoMLP
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


train_env_num = 32
test_env_num = 4
distribution_str = 'Beta'
device = torch.device('cpu')
log_dir = 'logs_py/humanoid/ppo/exp_0/'

actor_critic_args = {'input_size': 376 + 1, 'hidden_size': 64, 'action_size': 2 * 17}
soft_ppo_args = {
    'normalize_adv': True,
    'ppo_n_epoch': 10, 'ppo_n_mini_batches': 8
}
trainer_args = {
    'warm_up_steps': 0, 'n_plot_agents': 1,
    'log_dir': log_dir, 'recurrent': False
}
train_args = {
    'n_epoch': 5, 'n_steps_per_epoch': 500,
    'rollout_len': 128, 'n_tests_per_epoch': 100
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


def make_model():
    def make_ac():
        return ActorCriticTwoMLP(**actor_critic_args)
    model = AgentModel(
        make_ac, distribution_str,
        make_obs_normalizer=True,
        make_reward_normalizer=True,
        make_value_normalizer=True
    )
    return model


def main():
    create_log_dir(log_dir, __file__)

    train_env = make_vec_env(train_env_num)
    test_env = make_vec_env(test_env_num)

    model = make_model()
    model.share_memory()
    agent = PPO(model, device, **soft_ppo_args)

    trainer = OnPolicyTrainer(
        agent, train_env,
        **trainer_args,
        test_env=test_env
    )
    trainer.train(**train_args)

    train_env.close()
    test_env.close()


if __name__ == '__main__':
    main()
