# import sys
# import os

# sys.path.insert(1, (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

import gym
import torch
import argparse

from algorithms.ppo import PPO
from utils.init_env import init_env
from trainers.on_policy import OnPolicyTrainer


def parse_env(env_name):
    env = gym.make(env_name)
    image_env = False
    if len(env.observation_space.shape) == 3:
        image_env = True

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    else:
        action_size = env.action_space.shape[0]

    observation_size = 1
    for i in env.observation_space.shape:
        observation_size *= i

    env.close()
    return observation_size, action_size, image_env


def main(args):
    # init env
    train_env = init_env(args.env_name, args.train_env_num)
    test_env = init_env(args.env_name, args.test_env_num)

    # init agent
    observation_size, action_size, image_env = parse_env(args.env_name)
    device = torch.device(args.device)

    agent = PPO(
        image_env,
        observation_size, action_size, args.hidden_size, device,
        args.policy, args.normalize_adv, args.returns_estimator,
        args.learning_rate, args.gamma, args.entropy, args.clip_grad,
        gae_lambda=args.gae_lambda,
        ppo_epsilon=args.ppo_epsilon, ppo_n_epoch=args.ppo_n_epoch,
        ppo_mini_batch=args.ppo_mini_batch
    )

    # init and run trainer
    trainer = OnPolicyTrainer(agent, train_env, test_env, args.log_dir)
    trainer.train(args.n_epoch, args.n_step_per_epoch, args.rollout_len, args.n_tests_per_epoch)

    train_env.close()
    test_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)

    # env
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--train_env_num", type=int)
    parser.add_argument("--test_env_num", type=int)

    # nn
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--device", type=str)

    # policy & advantage
    parser.add_argument("--policy", type=str)
    parser.add_argument("--normalize_adv", action='store_true')
    parser.add_argument("--returns_estimator", type=str)

    # optimization
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--entropy", type=float)
    parser.add_argument("--clip_grad", type=float)
    parser.add_argument("--gae_lambda", type=float)

    # ppo
    parser.add_argument("--ppo_epsilon", type=float)
    parser.add_argument("--ppo_n_epoch", type=int)
    parser.add_argument("--ppo_mini_batch", type=int)

    # training
    parser.add_argument("--n_epoch", type=int)
    parser.add_argument("--n_step_per_epoch", type=int)
    parser.add_argument("--rollout_len", type=int)
    parser.add_argument("--n_tests_per_epoch", type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
