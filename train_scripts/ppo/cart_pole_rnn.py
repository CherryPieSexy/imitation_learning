import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.agent_model import AgentModel
from algorithms.nn.recurrent_encoders import init, CompositeRnnEncoder, OneLayerActorCritic
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


# WARNING: this experiment gives bad results.
train_env_num = 4
test_env_num = 4
distribution_str = 'Categorical'
device = torch.device('cpu')
log_dir = 'logs_py/cart_pole/ppo/exp_1_gru/'

input_size = 4
hidden_size = 16
action_size = 2

actor_critic_args = {'input_size': hidden_size, 'action_size': action_size}
ppo_args = {
    'learning_rate': 0.001,
    'ppo_n_epoch': 4, 'ppo_n_mini_batches': 4,
    'rollback_alpha': 0.1
}
trainer_args = {'log_dir': log_dir, 'recurrent': True}
train_args = {
    'n_epoch': 5, 'n_steps_per_epoch': 500,
    'rollout_len': 16, 'n_tests_per_epoch': 1,
}


def make_env():
    def make():
        env = gym.make('CartPole-v1')
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        return env
    return make


def make_vec_env(n_envs):
    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    return vec_env


def make_encoder():
    def make():
        gain = torch.nn.init.calculate_gain('tanh')
        enc_ = torch.nn.Sequential(
            init(torch.nn.Linear(input_size, hidden_size), gain=gain),
            torch.nn.Tanh()
        )
        return enc_
    return CompositeRnnEncoder(make, hidden_size, hidden_size)


def make_model():
    def make_ac():
        return OneLayerActorCritic(**actor_critic_args)
    model = AgentModel(
        make_ac, distribution_str,
        make_obs_encoder=make_encoder
    )
    return model


def main():
    create_log_dir(log_dir, __file__)

    train_env = make_vec_env(train_env_num)
    test_env = make_vec_env(test_env_num)

    model = make_model()
    agent = PPO(model, device, **ppo_args)

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
