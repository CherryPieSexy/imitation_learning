import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.agent_model import AgentModel
from algorithms.nn.conv_encoders import DeepConvEncoder
from algorithms.nn.actor_critic import ActorCriticTwoMLP
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


train_env_num = 16
test_env_num = 2
distribution_str = 'Beta'
device = torch.device('cpu')
log_dir = 'logs_py/car_racing/exp_0/'

input_size = 256
hidden_size = 100
action_size = 2 * 3

actor_critic_args = {
    'input_size': input_size, 'hidden_size': hidden_size, 'action_size': action_size
}
ppo_args = {
    'learning_rate': 5e-5, 'gamma': 0.995, 'clip_grad': 0.1,
    'rollback_alpha': 0.1,
    'ppo_n_epoch': 3, 'ppo_n_mini_batches': 8
}
trainer_args = {'log_dir': log_dir, 'recurrent': False}
train_args = {
    'n_epoch': 5, 'n_steps_per_epoch': 500,
    'rollout_len': 64, 'n_tests_per_epoch': 10
}


def make_env(env_id=0, n_envs=1):
    max_time = 1000

    def make():
        env = gym.make('CarRacing-v0', verbose=0)
        env = wrappers.ContinuousActionWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        env = wrappers.ImageEnvWrapper(env, x_size=96, y_size=96)
        env = wrappers.FrameStackWrapper(env, 4)
        env = wrappers.DeSyncWrapper(env, max_time, env_id, n_envs)
        return env
    return make


def make_vec_env(n_envs, train):
    vec_env = SubprocVecEnv([
        make_env(i if train else 0, n_envs)
        for i in range(n_envs)
    ])
    return vec_env


def make_model():
    def make_ac():
        return ActorCriticTwoMLP(**actor_critic_args)

    return AgentModel(
        make_ac, distribution_str,
        make_obs_encoder=DeepConvEncoder
    )


def main():
    create_log_dir(log_dir, __file__)

    train_env = make_vec_env(train_env_num, True)
    test_env = make_vec_env(test_env_num, False)

    model = make_model()
    agent = PPO(model, device, **ppo_args)

    trainer = OnPolicyTrainer(
        agent, train_env,
        **trainer_args,
        test_env=test_env,
    )
    trainer.train(**train_args)

    train_env.close()
    test_env.close()


if __name__ == '__main__':
    main()
