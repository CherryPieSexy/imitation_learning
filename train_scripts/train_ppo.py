import gym

from algorithms.ppo import PPO
from utils.vec_env import SubprocVecEnv
from utils.env_wrappers import OneHotWrapper
from trainers.on_policy import OnPolicyTrainer


# TODO: move this function somewhere in utils
def make_env(env_name):
    def _make_env():
        env = gym.make(env_name)
        env = OneHotWrapper(env)
        return env
    return _make_env


if __name__ == '__main__':
    agent = PPO(
        4, 2, 64, 'cpu',
        'GumbelSoftmax', True, 'gae',
        1e-3, 0.99, 1e-3, 100500,
        ppo_epsilon=0.1, ppo_n_epochs=3, ppo_mini_batch=15
    )
    train_env = SubprocVecEnv([make_env('CartPole-v1') for _ in range(4)])
    test_env = SubprocVecEnv([make_env('CartPole-v1') for _ in range(4)])
    trainer = OnPolicyTrainer(agent, train_env, test_env, '../logs/6/')
    trainer.train(5, 1000, 5, 10)

    train_env.close()
    test_env.close()
