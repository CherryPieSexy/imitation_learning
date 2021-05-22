import gym
import torch
import torch.multiprocessing as mp

import torch_rl.utils.env_wrappers as wrappers
from torch_rl.utils.utils import create_log_dir

from torch_rl.algorithms.nn.actor_critic import ActorCriticTwoMLP
from torch_rl.algorithms.nn.agent_model import AgentModel
from torch_rl.algorithms.optimizers.ppo import PPO

import torch_rl.algorithms.parallel as parallel


log_dir = 'logs_py/parallel/cart_pole/exp_2_ppo/'
device = torch.device('cpu')
recurrent = False

distribution_str = 'Categorical'

ac_args = {'input_size': 4, 'hidden_size': 16, 'action_size': 2}
ppo_args = {
    'learning_rate': 0.01, 'returns_estimator': '1-step',
    'ppo_n_epoch': 4, 'ppo_n_mini_batches': 4,
    'rollback_alpha': 0.1
}
train_args = {
    'train_env_num': 4, 'gamma': 0.99, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 500, 'rollout_len': 16}

run_test_process = True
render_test_env = True
test_process_act_deterministic = False


def make_env():
    def make():
        env = gym.make('CartPole-v1')
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        return env
    return make


def make_ac_model():
    def make_ac():
        return ActorCriticTwoMLP(**ac_args)
    model = AgentModel(device, make_ac, distribution_str)
    return model


def make_optimizer(model):
    return PPO(model, **ppo_args)


def main():
    create_log_dir(log_dir, __file__)
    parallel.run(
        log_dir, make_env, make_ac_model, render_test_env,
        make_optimizer, train_args, training_args,
        run_test_process=run_test_process,
        test_process_act_deterministic=test_process_act_deterministic
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
