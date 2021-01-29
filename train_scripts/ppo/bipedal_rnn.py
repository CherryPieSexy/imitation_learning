import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.agent_model import AgentModel
from algorithms.nn.recurrent_encoders import init, CompositeRnnEncoder, OneLayerActorCritic
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


# Achieves higher reward in less train-ops compared to feed forward model.
train_env_num = 16
test_env_num = 4
distribution_str = 'Beta'
device = torch.device('cpu')
log_dir = 'logs_py/bipedal/ppo/exp_1_gru/'

input_size = 24
gru_hidden_size = 64
action_size = 2 * 4

actor_critic_args = {'input_size': gru_hidden_size, 'action_size': action_size}
ppo_args = {
    'normalize_adv': True, 'entropy': 0.025,
    'ppo_n_epoch': 5, 'ppo_n_mini_batches': 4,
    'ppo_epsilon': 0.1
}
trainer_args = {'log_dir': log_dir, 'recurrent': True}
train_args = {
    'n_epoch': 5, 'n_steps_per_epoch': 500,
    'rollout_len': 32, 'n_tests_per_epoch': 5,
}


def make_env():
    def make():
        env = gym.make('BipedalWalker-v3')
        env = wrappers.ContinuousActionWrapper(env)
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
            init(torch.nn.Linear(input_size, gru_hidden_size), gain=gain),
            torch.nn.Tanh()
        )
        return enc_
    return CompositeRnnEncoder(make, gru_hidden_size, gru_hidden_size)


def make_model():
    def make_ac():
        return OneLayerActorCritic(**actor_critic_args)

    model = AgentModel(
        make_ac, distribution_str,
        make_obs_normalizer=True,
        make_obs_encoder=make_encoder,
        make_reward_normalizer=True,
        make_value_normalizer=True
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
