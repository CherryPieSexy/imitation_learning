import shutil

import yaml
import torch
import argparse

from utils.init_env import init_env
from utils.utils import create_log_dir
from algorithms.nn.actor_critic import init_actor_critic
from algorithms.agents.base_agent import AgentInference
from algorithms.agents.a2c import A2C
from algorithms.agents.ppo import PPO
from algorithms.agents.v_mpo import VMPO
from trainers.on_policy import OnPolicyTrainer


def main(args):
    # init env
    train_env_args, test_env_args = args['train_env_args'], args['test_env_args']
    train_env = init_env(**train_env_args)
    test_env = init_env(**test_env_args)

    # init net and agent
    device_online = torch.device(args['device_online'])
    device_train = torch.device(args['device_train'])

    nn_online = init_actor_critic(args['actor_critic_nn_type'], args['actor_critic_nn_args'])
    nn_train = init_actor_critic(args['actor_critic_nn_type'], args['actor_critic_nn_args'])

    nn_online.to(device_online)
    nn_train.to(device_train)

    policy = args['policy']
    policy_args = args['policy_args']

    agent_online = AgentInference(nn_online, device_online, policy, policy_args)

    train_agent_args = args['train_agent_args']
    agent_type = train_agent_args['agent_type']
    if agent_type == 'A2C':
        agent_class = A2C
    elif agent_type == 'PPO':
        agent_class = PPO
    elif agent_type == 'V-MPO':
        agent_class = VMPO
    else:
        raise ValueError(f'only A2C, PPO and V-MPO agents supported, provided {agent_type}')
    optimization_params = train_agent_args['optimization_params']
    additional_params = train_agent_args['additional_params']

    agent_train = agent_class(
        nn_train, device_train,
        policy, policy_args,
        normalize_adv=train_agent_args['normalize_advantage'],
        returns_estimator=train_agent_args['returns_estimator'],
        **optimization_params, **additional_params
    )

    # init and run trainer
    trainer_args = args['trainer_args']
    trainer = OnPolicyTrainer(
        agent_online, agent_train,
        train_env,
        **trainer_args,
        test_env=test_env,
        log_dir=args['log_dir']
    )

    if args['load_checkpoint'] is not None:
        trainer.load(args['load_checkpoint'])

    training_args = args['training_args']
    trainer.train(**training_args)

    train_env.close()
    test_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='path to yaml config', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args_ = parse_args()
    with open(args_.config) as f:
        config = yaml.safe_load(f)

    # create log-dir and copy config into it
    create_log_dir(config['log_dir'])
    shutil.copyfile(args_.config, config['log_dir'] + 'config.yaml')

    main(config)
