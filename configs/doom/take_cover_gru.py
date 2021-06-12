import torch
import torch.multiprocessing as mp

from cherry_rl.utils.image_wrapper import ImageEnvWrapper
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.doom_cnn import DoomCNN
from cherry_rl.algorithms.nn.recurrent_encoders import init, CompositeRnnEncoder, OneLayerActorCritic
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.ppo import PPO

import cherry_rl.algorithms.parallel as parallel

from configs.doom.doom_env import Doom

log_dir = 'logs/doom/take_cover/exp_0_ppo/'
device = torch.device('cuda')
recurrent = True

cnn_output_size = 2304
gru_hidden_size = 512
action_size = 2
distribution_str = 'Bernoulli'

gamma = 0.99
train_env_num = 32
rollout_len = 32


ac_args = {'hidden_size': gru_hidden_size, 'action_size': action_size}
ppo_args = {
    'normalize_adv': True, 'clip_grad': 0.5,
    'returns_estimator': 'v-trace',
    'ppo_n_epoch': 3, 'ppo_n_mini_batches': 4
}

train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 20, 'n_steps_per_epoch': 5000, 'rollout_len': rollout_len}

run_test_process = True
render_test_env = False
test_process_act_deterministic = False


def make_env(render=False):
    def make():
        env = Doom('take_cover', set_window_visible=render, action_repeat=4)
        env = ImageEnvWrapper(env, convert_to_gray=False, x_size=128, y_size=72)
        return env
    return make


def make_encoder():
    def make():
        gain = torch.nn.init.calculate_gain('relu')
        encoder = torch.nn.Sequential(
            DoomCNN(input_channels=3),
            init(torch.nn.Linear(cnn_output_size, gru_hidden_size), gain=gain),
            torch.nn.ReLU()
        )
        return encoder
    return CompositeRnnEncoder(make, gru_hidden_size, gru_hidden_size)


def make_ac_model():
    def make_ac():
        return OneLayerActorCritic(**ac_args)

    model = AgentModel(
        device, make_ac, distribution_str,
        make_obs_encoder=make_encoder,
        reward_scaler_size=1, value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    return PPO(model, **ppo_args)


def main():
    create_log_dir(log_dir, __file__)
    parallel.run(
        log_dir, make_env, make_ac_model, render_test_env,
        make_optimizer, train_args, training_args,
        run_test_process=run_test_process
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
