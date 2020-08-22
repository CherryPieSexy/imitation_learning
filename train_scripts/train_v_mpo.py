import os
import gym
import json
import torch
import argparse

from algorithms.policy_gradient import AgentInference
from algorithms.v_mpo import VMPO
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
        json.dump(vars(args), f, indent=4)

    # init env
    train_env = init_env(
        args.env_name, args.train_env_num,
        action_repeat=args.action_repeat,
        die_penalty=args.die_penalty,
        max_len=args.max_episode_len
    )
    test_env = init_env(args.env_name, args.test_env_num, action_repeat=args.action_repeat)

    # init agent
    observation_size, action_size, image_env = parse_env(args.env_name)
    device_online = torch.device(args.device_online)
    device_train = torch.device(args.device_train)

    agent_online = AgentInference(
        image_env,
        observation_size, action_size, args.hidden_size, device_online,
        args.policy
    )
    agent_train = VMPO(
        image_env,
        observation_size, action_size, args.hidden_size, device_train,
        args.policy, args.normalize_adv, args.returns_estimator,
        args.learning_rate, args.gamma, args.entropy, args.clip_grad,
        eps_eta_range=args.eps_eta_range, eps_mu_range=args.eps_mu_range,
        eps_sigma_range=args.eps_sigma_range
    )
    agent_online.load_state_dict(agent_train.state_dict())

    trainer = OnPolicyTrainer(
        agent_online, agent_train,
        args.update_period, args.return_pi,
        train_env, test_env,
        args.normalize_obs, args.normalize_reward,
        args.obs_clip, args.reward_clip,
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
    parser.add_argument("--action_repeat", type=int)
    parser.add_argument("--die_penalty", type=int, default=0)
    parser.add_argument("--max_episode_len", type=int, default=0)
    parser.add_argument("--train_env_num", type=int)
    parser.add_argument("--test_env_num", type=int)

    # nn
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--device_online", type=str)
    parser.add_argument("--device_train", type=str)

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

    # v-mpo
    parser.add_argument("--eps_eta_range", nargs="+", type=float, default=[0.01, 0.01])
    parser.add_argument("--eps_mu_range", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--eps_sigma_range", nargs="+", type=float, default=[5e-6, 5e-5])

    # trainer
    parser.add_argument("--update_period", type=int)
    parser.add_argument("--return_pi", action='store_true')
    parser.add_argument("--normalize_obs", action='store_true')
    parser.add_argument("--normalize_reward", action='store_true')
    parser.add_argument("--obs_clip", type=float, default=-float('inf'))
    parser.add_argument("--reward_clip", type=float, default=float('inf'))

    # training
    parser.add_argument("--n_epoch", type=int)
    parser.add_argument("--n_step_per_epoch", type=int)
    parser.add_argument("--rollout_len", type=int)
    parser.add_argument("--n_tests_per_epoch", type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
