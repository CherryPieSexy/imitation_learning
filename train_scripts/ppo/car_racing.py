import torch

from utils.utils import create_log_dir
from utils.init_env import init_env
from algorithms.nn.conv_encoders import ConvEncoder, TwoLayerActorCritic
from algorithms.normalization import RunningMeanStd
from algorithms.agents.base_agent import AgentInference
from algorithms.agents.ppg import PPG
from trainers.on_policy import OnPolicyTrainer


train_env_num = 8
test_env_num = 2
distribution = 'Beta'
device = torch.device('cpu')
log_dir = 'logs_py/car_racing/exp_0/'

env_args = {
    'env_type': 'gym', 'env_name': 'CarRacing-v0', 'env_args': {'verbose': 0},
    'action_repeat': 2, 'frame_stack': 4, 'image_args': {'x_size': 96, 'y_size': 96}
}

actor_critic_args = {'action_size': 3, 'distribution': distribution}
agent_train_args = {
    'learning_rate': 5e-5, 'gamma': 0.995, 'clip_grad': 0.1,
    'rollback_alpha': 0.1,
    'ppo_n_epoch': None, 'ppo_n_mini_batches': 8
}

trainer_args = {'warm_up_steps': 10, 'n_plot_agents': 1, 'log_dir': log_dir}
train_args = {
    'n_epoch': 20, 'n_steps_per_epoch': 50,
    'rollout_len': 128, 'n_tests_per_epoch': 10
}


def make_agent_online():
    actor_critic_online = TwoLayerActorCritic(**actor_critic_args)
    obs_encoder_online = ConvEncoder()
    obs_normalizer = RunningMeanStd()
    agent = AgentInference(
        actor_critic_online, device, distribution,
        obs_encoder=obs_encoder_online,
        obs_normalizer=obs_normalizer,
    )
    return agent


def make_agent_train():
    actor_critic_train = TwoLayerActorCritic(**actor_critic_args)
    obs_encoder_train = ConvEncoder()
    obs_normalizer = RunningMeanStd()
    agent = PPG(
        actor_critic_train, device, distribution,
        obs_encoder=obs_encoder_train,
        obs_normalizer=obs_normalizer,
        **agent_train_args
    )
    return agent


def main():
    create_log_dir(log_dir, __file__)

    train_env = init_env(**env_args, env_num=train_env_num)
    test_env = init_env(**env_args, env_num=test_env_num)

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
