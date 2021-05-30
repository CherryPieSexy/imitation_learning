# imports
import gym
import torch
import torch.multiprocessing as mp

import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.actor_critic import ActorCriticTwoMLP
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.ppo import PPO

import cherry_rl.algorithms.parallel as parallel


log_dir = 'logs/cart_pole/exp_3_ppo_annotated/'  # tensorboard logs and checkpoints will be stored in 'log_dir'
device = torch.device('cpu')
recurrent = False

distribution_str = 'Categorical'  # policy distribution for discrete action space

ac_args = {'input_size': 4, 'hidden_size': 16, 'action_size': 2}  # actor-critic mlp args
# PPO optimizer args
ppo_args = {
    'learning_rate': 0.01,
    'returns_estimator': '1-step',
    'ppo_n_epoch': 4,         # PPO will train on every rollout 4 times
    'ppo_n_mini_batches': 4,  # every rollout will be split into 4 mini-batches
    'rollback_alpha': 0.1
}
train_args = {
    'train_env_num': 4,  # number of environments working in parallel to collect rollouts
    'gamma': 0.99, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {
    'n_epoch': 10,
    'n_steps_per_epoch': 500,  # number of collected rollouts per one training epoch, i.e. epoch size
    'rollout_len': 16
}

run_test_process = True  # if set to True then additional test process will be run along with training
render_test_env = True  # if set to True then policy playing in the test environment will be rendered
test_process_act_deterministic = False


# function to create environment
# any environment with gym interface will work correctly
def make_env():
    def make():
        env = gym.make('CartPole-v1')
        # special wrapper to do correct rendering with action-repeat (frame-skip)
        # it change step method to have 'render' kwarg and used in every experiment
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        return env
    return make


# function to create actor-critic nn
# any nn with same interface as ActorCriticTwoMLP will work correctly
def make_ac_model():
    def make_ac():
        return ActorCriticTwoMLP(**ac_args)
    model = AgentModel(device, make_ac, distribution_str)
    return model


# function to create optimizer, i.e. reinforcement learning algorithm
def make_optimizer(model):
    return PPO(model, **ppo_args)


# main function, it creates all necessary processes and perform training
# look into 'parallel' for parallelism scheme and details
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
