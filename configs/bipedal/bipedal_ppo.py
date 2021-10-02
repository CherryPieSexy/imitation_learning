import gym
import torch
import torch.multiprocessing as mp

import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.actor_critic import ActorCriticTwoMLP
from cherry_rl.algorithms.distributions import Beta
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.rl.ppo import PPO

import cherry_rl.algorithms.parallel as parallel


log_dir = 'logs/bipedal/exp_1_ppo/'
device = torch.device('cpu')
recurrent = False

observation_size = 24
hidden_size = 64
action_size = 4
distribution = Beta

gamma = 0.99
train_env_num = 32
rollout_len = 16

ac_args = {'input_size': observation_size, 'hidden_size': hidden_size, 'action_size': action_size * 2}
ppo_args = {
    # 'normalize_adv': True,
    'returns_estimator': 'v-trace',
    'ppo_n_epoch': 8, 'ppo_n_mini_batches': 4,
    'ppo_epsilon': 0.1
}
train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 15, 'n_steps_per_epoch': 500, 'rollout_len': rollout_len}

run_test_process = True
render_test_env = True
test_process_act_deterministic = True


def make_env():
    def make():
        env = gym.make('BipedalWalker-v3')
        env = wrappers.ContinuousActionWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        return env
    return make


def make_ac_model(ac_device):
    def make_ac():
        return ActorCriticTwoMLP(**ac_args)
    model = AgentModel(
        ac_device, make_ac, distribution,
        obs_normalizer_size=observation_size,
        reward_normalizer_size=1,
        value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    return PPO(model, **ppo_args)


def main():
    create_log_dir(log_dir, __file__)
    parallel.run(
        log_dir, make_env, make_ac_model, device,
        make_optimizer, train_args, training_args,
        run_test_process=run_test_process,
        render_test_env=render_test_env,
        test_process_act_deterministic=test_process_act_deterministic
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
