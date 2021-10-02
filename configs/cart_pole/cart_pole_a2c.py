import gym
import torch
import torch.multiprocessing as mp

import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.actor_critic import ActorCriticTwoMLP
from cherry_rl.algorithms.distributions import Categorical
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.rl.a2c import A2C

import cherry_rl.algorithms.parallel as parallel


# WARNING: this config can't give stable results. It may converge in 1 or 10 minutes.
# Possible reason: parallel training and rollout gathering and off-policy updates.
# Sequential A2C should converge in ~20 seconds.
log_dir = 'logs/cart_pole/exp_1_a2c_2/'
device = torch.device('cpu')
recurrent = False

distribution = Categorical

ac_args = {'input_size': 4, 'hidden_size': 16, 'action_size': 2}
a2c_args = {'learning_rate': 0.01, 'returns_estimator': '1-step'}
train_args = {
    'train_env_num': 4, 'gamma': 0.99, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 5, 'n_steps_per_epoch': 500, 'rollout_len': 16}

run_test_process = True
render_test_env = True
test_process_act_deterministic = False


def make_env():
    def make():
        env = gym.make('CartPole-v1')
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        return env
    return make


def make_ac_model(ac_device):
    def make_ac():
        return ActorCriticTwoMLP(**ac_args)
    model = AgentModel(ac_device, make_ac, distribution)
    return model


def make_optimizer(model):
    return A2C(model, **a2c_args)


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
