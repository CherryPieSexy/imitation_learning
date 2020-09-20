import shutil

import yaml
import torch
from torch.utils.data import DataLoader
import argparse

from utils.init_env import init_env
from utils.utils import create_log_dir
from algorithms.agents.bc import BehaviorCloning, BCDataSet
from trainers.behavior_cloning import BehaviorCloningTrainer


def main(args):
    test_env_args = args['test_env_args']
    test_env = init_env(**test_env_args)

    # init agent
    device_train = torch.device(args['device_train'])
    nn_train = init_actor_critic(args['actor_critic_nn_type'], args['actor_critic_nn_args'])
    agent = BehaviorCloning(
        image_env,
        observation_size, action_size, args.hidden_size,
        device, args.policy,
        lr=args.learning_rate, clip_grad=args.clip_grad,
        normalize_adv=None,
        returns_estimator=None,
        gamma=None, entropy=None
    )

    # init demo buffer
    demo_data = BCDataSet(args.demo_file)
    demo_buffer = DataLoader(demo_data, args.batch_size, shuffle=True)

    # init trainer and train
    trainer = BehaviorCloningTrainer(agent, test_env, demo_buffer, args.log_dir)
    if args.load_checkpoint is not None:
        trainer.load(args.load_checkpoint)
    trainer.train(args.n_epoch, args.n_tests_per_epoch)
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
