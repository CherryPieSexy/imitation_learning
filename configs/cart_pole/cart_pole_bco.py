import gym
import torch
import torch.multiprocessing as mp

import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.demo_buffer import TransitionsDemoBuffer
from cherry_rl.algorithms.nn.actor_critic import ActorCriticTwoMLP
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.nn.inverse_dynamics_model import InverseDynamicsModel
from cherry_rl.algorithms.optimizers.d_kl_policy_optimizer import PolicyOptimizer
from cherry_rl.algorithms.optimizers.inverse_dynamics_optimizer import InverseDynamicsOptimizer
from cherry_rl.algorithms.optimizers.bco import BCO

import cherry_rl.algorithms.parallel as parallel


log_dir = 'logs/cart_pole/exp_5_bco/'
device = torch.device('cpu')
recurrent = False

distribution_str = 'Categorical'

demo_file = 'configs/cart_pole/cart_pole_demo_10_ep.pickle'

ac_args = {'input_size': 4, 'hidden_size': 16, 'action_size': 2}
idm_args = {
    'observation_size': 4, 'hidden_size': 16,
    'action_size': 2, 'distribution_str': distribution_str
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
    policy_optimizer = PolicyOptimizer(model)
    idm = InverseDynamicsModel(**idm_args)
    idm_optimizer = InverseDynamicsOptimizer(idm)
    demo_buffer = TransitionsDemoBuffer(demo_file, 4 * 16)
    optimizer = BCO(policy_optimizer, idm_optimizer, demo_buffer)
    return optimizer


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