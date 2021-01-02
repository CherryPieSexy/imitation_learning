import torch
from torch.utils.data import DataLoader

from utils.utils import create_log_dir
from utils.init_env import init_env
from algorithms.nn.actor_critic import ActorCriticTwoMLP
from algorithms.agents.bc import BehaviorCloningAgent, BCDataSet
from trainers.behavior_cloning import BehaviorCloningTrainer


test_env_num = 4
distribution = 'Categorical'
device = torch.device('cpu')
log_dir = 'logs_py/cart_pole/bc_10_episodes/'

demo_file = 'demo_files/cart_pole_demo_10.pickle'
batch_size = 32

env_args = {'env_type': 'gym', 'env_name': 'CartPole-v1', 'env_args': dict()}
actor_critic_args = {
    'observation_size': 4, 'hidden_size': 16, 'action_size': 2,
    'distribution': distribution
}
agent_train_args = {'learning_rate': 0.01, 'loss_type': 'log_prob'}
trainer_args = {'n_plot_agents': 1, 'log_dir': log_dir}
train_args = {'n_epoch': 10, 'n_tests_per_epoch': 100}


def make_agent_online():
    # test.py will use 'config.make_agent_online' method to create agent
    actor_critic_train = ActorCriticTwoMLP(**actor_critic_args)
    agent = BehaviorCloningAgent(
        actor_critic_train, device, distribution,
        **agent_train_args
    )
    return agent


def main():
    create_log_dir(log_dir, __file__)

    test_env = init_env(**env_args, env_num=test_env_num)
    demo_data = BCDataSet(demo_file)
    demo_buffer = DataLoader(demo_data, batch_size=batch_size, shuffle=True)
    agent = make_agent_online()

    trainer = BehaviorCloningTrainer(
        agent, demo_buffer,
        test_env=test_env,
        log_dir=log_dir
    )
    trainer.train(**train_args)

    test_env.close()


if __name__ == '__main__':
    main()
