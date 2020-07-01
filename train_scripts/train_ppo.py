import os
import gym
import json
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
    # tensorboard logs saved in 'log_dir/tb/', checkpoints in 'log_dir/checkpoints'
    try:
        os.mkdir(args.log_dir)
        os.mkdir(args.log_dir + 'tb')
        os.mkdir(args.log_dir + 'checkpoints')
    except FileExistsError:
        print('log_dir already exists')

    # save training config
    with open(args.log_dir + 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # init env
    train_env = init_env(args.env_name, args.train_env_num, action_repeat=args.action_repeat)
    test_env = init_env(args.env_name, args.test_env_num, action_repeat=args.action_repeat)

    # init agent
    observation_size, action_size, image_env = parse_env(args.env_name)
    device = torch.device(args.device)

    agent = PPO(
        image_env,
        observation_size, action_size, args.hidden_size, device,
        args.policy, args.normalize_adv, args.returns_estimator,
        args.learning_rate, args.gamma, args.entropy, args.clip_grad,
        gae_lambda=args.gae_lambda,
        ppo_epsilon=args.ppo_epsilon,
        use_ppo_value_loss=args.ppo_value_loss,
        rollback_alpha=args.rollback_alpha,
        recompute_advantage=args.recompute_advantage,
        ppo_n_epoch=args.ppo_n_epoch,
        ppo_n_mini_batches=args.ppo_n_mini_batches,
    )

    # init and run trainer
    trainer = OnPolicyTrainer(
        agent, train_env, test_env,
        args.normalize_obs, args.normalize_reward,
        args.reward_clip_min, args.reward_clip_max,
        args.log_dir
    )
    if args.load_checkpoint is not None:
        trainer.load(args.load_checkpoint)
    trainer.train(args.n_epoch, args.n_step_per_epoch, args.rollout_len, args.n_tests_per_epoch)

    train_env.close()
    test_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--load_checkpoint", type=str)

    # env
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--normalize_obs", action='store_true', default=True)
    parser.add_argument("--normalize_reward", action='store_true', default=True)
    parser.add_argument("--reward_clip_min", type=float, default=-float('inf'))
    parser.add_argument("--reward_clip_max", type=float, default=float('inf'))
    parser.add_argument("--action_repeat", type=int)
    parser.add_argument("--train_env_num", type=int)
    parser.add_argument("--test_env_num", type=int)

    # nn
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--device", type=str)

    # policy & advantage
    parser.add_argument("--policy", type=str)
    parser.add_argument("--normalize_adv", action='store_true')
    parser.add_argument("--returns_estimator", type=str, default='gae')

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy", type=float, default=1e-3)
    parser.add_argument("--clip_grad", type=float, default=0.5)
    parser.add_argument("--gae_lambda", type=float, default=0.9)

    # ppo
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)
    parser.add_argument("--ppo_value_loss", action='store_true')
    parser.add_argument("--rollback_alpha", type=float, default=0.0)
    parser.add_argument("--recompute_advantage", action='store_true')
    parser.add_argument("--ppo_n_epoch", type=int)
    parser.add_argument("--ppo_n_mini_batches", type=int)

    # training
    parser.add_argument("--n_epoch", type=int)
    parser.add_argument("--n_step_per_epoch", type=int)
    parser.add_argument("--rollout_len", type=int)
    parser.add_argument("--n_tests_per_epoch", type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
