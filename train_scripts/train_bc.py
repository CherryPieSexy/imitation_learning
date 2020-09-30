import shutil

import yaml
import torch
from torch.utils.data import DataLoader
import argparse

from utils.init_env import init_env
from utils.utils import create_log_dir
from algorithms.nn.actor_critic import init_actor_critic
from algorithms.agents.bc import BehaviorCloning, BCDataSet
from trainers.behavior_cloning import BehaviorCloningTrainer


def main(args):
    # init env
    test_env_args = args['test_env_args']
    test_env = init_env(**test_env_args)

    # init net and agent
    device_train = torch.device(args['device_train'])
    nn_train = init_actor_critic(args['actor_critic_nn_type'], args['actor_critic_nn_args'])
    nn_train.to(device_train)
    policy = args['policy']
    policy_args = args['policy_args']

    train_agent_args = args['train_agent_args']
    optimization_params = train_agent_args['optimization_params']
    additional_params = train_agent_args['additional_params']

    agent = BehaviorCloning(
        nn_train, device_train, policy, policy_args,
        **optimization_params, **additional_params
    )

    # init demo buffer
    demo_data = BCDataSet(args['demo_file'])
    demo_buffer = DataLoader(demo_data, args['batch_size'], shuffle=True)

    # init trainer and train
    trainer = BehaviorCloningTrainer(
        agent, demo_buffer,
        test_env=test_env,
        log_dir=args['log_dir']
    )
    if args['load_checkpoint'] is not None:
        trainer.load(args.load_checkpoint)

    training_args = args['training_args']
    trainer.train(**training_args)

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
