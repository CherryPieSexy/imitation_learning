import gym
import torch

import utils.env_wrappers as wrappers
from utils.vec_env import SubprocVecEnv
from utils.utils import create_log_dir
from algorithms.nn.conv_encoders import DeepConvEncoder, TwoLayerActorCritic
from algorithms.agents.base_agent import AgentInference
from algorithms.agents.ppg import PPO
from trainers.on_policy import OnPolicyTrainer


train_env_num = 8
test_env_num = 2
distribution = 'Beta'
device = torch.device('cpu')
log_dir = 'logs_py/car_racing/exp_1/'

actor_critic_args = {
    'input_size': 256 + 1, 'hidden_size': 100, 'action_size': 3,
    'distribution': distribution
}
agent_train_args = {
    'learning_rate': 5e-5, 'gamma': 0.995, 'clip_grad': 0.1,
    'rollback_alpha': 0.1,
    'ppo_n_epoch': 3, 'ppo_n_mini_batches': 8
}

trainer_args = {'n_plot_agents': 1, 'log_dir': log_dir}
train_args = {
    'n_epoch': 20, 'n_steps_per_epoch': 50,
    'rollout_len': 128, 'n_tests_per_epoch': 10
}


def make_env(env_id=0, n_envs=1):
    def make():
        env = gym.make('CarRacing-v0', verbose=0)
        env = wrappers.ContinuousActionWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env, 2)
        env = wrappers.ImageEnvWrapper(env, x_size=96, y_size=96)
        env = wrappers.FrameStackWrapper(env, 4)
        env = wrappers.DeSyncWrapper(env, 500, env_id, n_envs)
        env = wrappers.TimeStepWrapper(env, 500)
        return env
    return make


def make_vec_env(n_envs, train):
    vec_env = SubprocVecEnv([
        make_env(i if train else 0, n_envs)
        for i in range(n_envs)
    ])
    return vec_env


def make_agent_online():
    actor_critic_online = TwoLayerActorCritic(**actor_critic_args)
    obs_encoder_online = DeepConvEncoder()
    agent = AgentInference(
        actor_critic_online, device, distribution,
        obs_encoder=obs_encoder_online
    )
    return agent


def make_agent_train():
    actor_critic_train = TwoLayerActorCritic(**actor_critic_args)
    obs_encoder_train = DeepConvEncoder()
    agent = PPO(
        actor_critic_train, device, distribution,
        obs_encoder=obs_encoder_train,
        **agent_train_args
    )
    return agent


def main():
    create_log_dir(log_dir, __file__)

    train_env = make_vec_env(train_env_num, True)
    test_env = make_vec_env(test_env_num, False)

    agent_online = make_agent_online()
    agent_train = make_agent_train()

    agent_online.load_state_dict(agent_train.state_dict())

    trainer = OnPolicyTrainer(
        agent_online, agent_train, train_env,
        **trainer_args,
        test_env=test_env,
    )
    trainer.train(**train_args)

    train_env.close()
    test_env.close()


if __name__ == '__main__':
    main()
