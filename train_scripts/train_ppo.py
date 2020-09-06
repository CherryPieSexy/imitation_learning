import torch
import argparse

from utils.init_env import init_env
from utils.utils import create_log_dir
from algorithms.nn import ActorCriticTwoMLP, ActorCriticCNN, ActorCriticDeepCNN
from algorithms.policy_gradient import AgentInference
from algorithms.agents.ppo import PPO
from trainers.on_policy import OnPolicyTrainer


def main(args):
    observation_size, action_size, image_env = create_log_dir(args)

    # init env
    train_env = init_env(
        args.env_name, args.train_env_num,
        action_repeat=args.action_repeat,
        die_penalty=args.die_penalty
    )
    test_env = init_env(args.env_name, args.test_env_num, action_repeat=args.action_repeat)

    # init net and agent
    device_online = torch.device(args.device_online)
    device_train = torch.device(args.device_train)

    if image_env:
        # if args.deep_cnn:
        if True:
            nn_online = ActorCriticDeepCNN(action_size, args.policy)
            nn_train = ActorCriticDeepCNN(action_size, args.policy)
        else:
            nn_online = ActorCriticCNN(action_size, args.policy)
            nn_train = ActorCriticCNN(action_size, args.policy)
    else:
        nn_online = ActorCriticTwoMLP(observation_size, action_size, args.hidden_size, args.policy)
        nn_train = ActorCriticTwoMLP(observation_size, action_size, args.hidden_size, args.policy)

    nn_online.to(device_online)
    nn_train.to(device_train)

    agent_online = AgentInference(nn_online, device_online, args.policy)

    agent_train = PPO(
        nn_train, device_train,
        args.policy, args.normalize_adv, args.returns_estimator,
        args.learning_rate, args.gamma, args.entropy, args.clip_grad,
        image_augmentation_alpha=args.image_aug_alpha,
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
        agent_online, agent_train, args.update_period, False,
        train_env,
        args.normalize_obs, args.scale_reward, args.normalize_reward,
        args.obs_clip, args.reward_clip,
        test_env=test_env,
        log_dir=args.log_dir
    )
    if args.load_checkpoint is not None:
        trainer.load(args.load_checkpoint)
    trainer.train(args.n_epoch, args.n_step_per_epoch, args.rollout_len, args.n_tests_per_epoch)

    train_env.close()
    test_env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', help='directory where logs will be stored', type=str)
    parser.add_argument("--load_checkpoint", help='checkpoint name to load before training', type=str)

    # env
    parser.add_argument('--env_name', help='name of the environment to train on', type=str)
    parser.add_argument(
        '--action_repeat',
        help='number of consecutive action executions in environment, '
             '1 means that agent execute 1 action per environment step, default 1',
        type=int, default=1
    )
    parser.add_argument(
        '--die_penalty',
        help='add \'die_penalty\' into reward if episode ended earlier than time-limit',
        type=int, default=0
    )
    parser.add_argument('--train_env_num', help='number of training environments run in parallel', type=int)
    parser.add_argument('--test_env_num', help='number of testing environments run in parallel', type=int)

    # nn
    parser.add_argument('--hidden_size', help='hidden size for MLP networks', type=int, default=0)
    parser.add_argument(
        '--device_online',
        help='device for NN which collects data from environments, must be \'cpu\' or \'cuda\'',
        type=str
    )
    parser.add_argument(
        '--device_train',
        help='device for NN which performs learning, must be \'cpu\' or \'cuda\'', type=str
    )
    parser.add_argument(
        '--image_aug_alpha',
        help='weight of image augmentation loss (D_KL(pi||pi_aug))',
        type=float, default=0.0
    )

    # policy & advantage
    parser.add_argument(
        '--policy',
        help='policy distribution. One from: '
             'Categorical, GumbelSoftmax, Beta, WideBeta, Normal, TanhNormal',
        type=str
    )
    parser.add_argument(
        '--normalize_adv',
        help='if True then advantage will be normalized over batch dimension. '
             'Normalized advantage is only used for policy update.',
        action='store_true'
    )
    parser.add_argument(
        '--returns_estimator',
        help='type of returns estimator, must be one from \'1-step\', \'n-step\', \'gae\', default \'gae\'',
        type=str, default='gae'
    )

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy', type=float, default=1e-3)
    parser.add_argument('--clip_grad', type=float, default=0.5)
    parser.add_argument('--gae_lambda', type=float, default=0.9)

    # ppo
    parser.add_argument('--ppo_epsilon', help='PPO clipping parameter, default=0.2', type=float, default=0.2)
    parser.add_argument(
        '--ppo_value_loss',
        help='if True then clipping value loss will be used, default False',
        action='store_true'
    )
    parser.add_argument(
        '--rollback_alpha',
        help='if > 0 then rollback loss with rb-alpha will be used, default 0.0',
        type=float, default=0.0
    )
    parser.add_argument(
        '--recompute_advantage',
        help='if True, then Advantage will be recomputed after each PPO train-op. default False',
        action='store_true'
    )
    parser.add_argument(
        '--ppo_n_epoch',
        help='number of PPO updates per one data rollout',
        type=int
    )
    parser.add_argument(
        '--ppo_n_mini_batches',
        help='number of mini-batches on which one rollout is split',
        type=int
    )

    # trainer
    parser.add_argument(
        '--normalize_obs',
        help='if True, then observations will be normalized by running mean and variance, '
             'only for non-image observations, default False',
        action='store_true'
    )
    parser.add_argument(
        '--normalize_reward',
        help='if True, then rewards will be normalized by running mean and variance, default False',
        action='store_true'
    )
    parser.add_argument(
        '--scale_reward',
        help='if True, then rewards will be normalized by running mean and variance, default False',
        action='store_true'
    )
    parser.add_argument('--obs_clip', type=float, default=-float('inf'))
    parser.add_argument('--reward_clip', type=float, default=float('inf'))
    parser.add_argument('--update_period', type=int, default=1)

    # training
    parser.add_argument('--n_epoch', help='number of training epoch', type=int)
    parser.add_argument('--n_step_per_epoch', help='number of rollout gathered per one epoch', type=int)
    parser.add_argument('--rollout_len', help='number of transitions in one rollout', type=int)
    parser.add_argument('--n_tests_per_epoch', help='number of test episodes at the end of each epoch', type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)
