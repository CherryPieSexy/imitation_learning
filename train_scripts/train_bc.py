import torch
from torch.utils.data import DataLoader
import argparse

from utils.init_env import init_env
from utils.utils import create_log_dir
from algorithms.bc import BehaviorCloning, BCDataSet
from trainers.behavior_cloning import BehaviorCloningTrainer


def main(args):
    observation_size, action_size, image_env = create_log_dir(args)
    test_env = init_env(args.env_name, args.test_env_num, action_repeat=args.action_repeat)

    # init agent
    device = torch.device(args.device)
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
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--load_checkpoint", type=str)

    # env
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--action_repeat", type=int)
    parser.add_argument("--die_penalty", type=int, default=0)
    parser.add_argument("--max_episode_len", type=int, default=0)
    parser.add_argument("--test_env_num", type=int)

    # demo buffer
    parser.add_argument("--demo_file", type=str)
    parser.add_argument("--batch_size", type=int)

    # nn & policy
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--policy", type=str)

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--clip_grad", type=float, default=0.5)

    # training
    parser.add_argument("--n_epoch", type=int)
    parser.add_argument("--n_tests_per_epoch", type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
