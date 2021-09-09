from functools import partial

import torch
import torch.multiprocessing as mp

from cherry_rl.utils.image_wrapper import ImageEnvWrapper
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.doom_cnn import DoomCNN
from cherry_rl.algorithms.nn.actor_critic import ActorCriticTwoMLP
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.rl.ppo import PPO

import cherry_rl.algorithms.parallel as parallel

from configs.doom.doom_env import Doom

log_dir = 'logs/doom/defend_the_line/exp_0_ppo/'
device = torch.device('cuda')
recurrent = False

obs_size = 2304
hidden_size = 512
action_size = 3
distribution_str = 'Bernoulli'

gamma = 0.99
train_env_num = 32
rollout_len = 32


ac_args = {
    'input_size': obs_size, 'hidden_size': hidden_size, 'action_size': action_size,
    'n_layers': 2, 'activation_str': 'relu'
}
ppo_args = {
    'normalize_adv': True,
    'returns_estimator': 'gae',
    'ppo_n_epoch': 8, 'ppo_n_mini_batches': 4
}

train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 500, 'rollout_len': rollout_len}

run_test_process = False
render_test_env = True
test_process_act_deterministic = False


def make_env(render=True):
    def make():
        env = Doom('defend_the_line', set_window_visible=render, action_repeat=4)
        env = ImageEnvWrapper(env, convert_to_gray=False, x_size=128, y_size=72)
        return env
    return make


def make_ac_model(ac_device):
    def make_ac():
        return ActorCriticTwoMLP(**ac_args)

    def make_cnn():
        return DoomCNN(3)

    model = AgentModel(
        ac_device, make_ac, distribution_str,
        make_obs_encoder=make_cnn,
        reward_scaler_size=1, value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    return PPO(model, **ppo_args)


def main():
    create_log_dir(log_dir, __file__)
    parallel.run(
        log_dir, partial(make_env, render=False), make_ac_model, device,
        make_optimizer, train_args, training_args,
        run_test_process=run_test_process,
        render_test_env=render_test_env,
        test_process_act_deterministic=test_process_act_deterministic
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
