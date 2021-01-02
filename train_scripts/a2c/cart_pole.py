import torch

from utils.utils import create_log_dir
from utils.init_env import init_env
from algorithms.nn.actor_critic import ActorCriticTwoMLP
from algorithms.agents.base_agent import AgentInference
from algorithms.agents.a2c import A2C
from trainers.on_policy import OnPolicyTrainer


train_env_num = 4
test_env_num = 4
distribution = 'Categorical'
device = torch.device('cpu')
log_dir = 'logs_py/cart_pole/a2c_exp_0/'

env_args = {'env_type': 'gym', 'env_name': 'CartPole-v1', 'env_args': dict()}
actor_critic_args = {
    'observation_size': 4, 'hidden_size': 16, 'action_size': 2,
    'distribution': distribution
}
agent_train_args = {'learning_rate': 0.01, 'returns_estimator': '1-step'}
trainer_args = {'warm_up_steps': 10, 'n_plot_agents': 1, 'log_dir': log_dir}
train_args = {
    'n_epoch': 10, 'n_steps_per_epoch': 100,
    'rollout_len': 16, 'n_tests_per_epoch': 100
}


def make_agent_online():
    actor_critic_online = ActorCriticTwoMLP(**actor_critic_args)
    agent = AgentInference(actor_critic_online, device, distribution)
    return agent


def make_agent_train():
    actor_critic_train = ActorCriticTwoMLP(**actor_critic_args)
    agent = A2C(
        actor_critic_train, device, distribution,
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
        test_env=test_env
    )
    trainer.train(**train_args)

    train_env.close()
    test_env.close()


if __name__ == '__main__':
    main()
