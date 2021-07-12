import torch
import torch.multiprocessing as mp

from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.recurrent_encoders import OneLayerActorCritic
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.rl.ppo import PPO

import cherry_rl.algorithms.parallel as parallel

from configs.mario.encoder import Encoder
from configs.mario.env import make_mario_env


world, level = 1, 3
log_dir = f'logs/mario/world-{world}-level-{level}/'
device = torch.device('cuda')
recurrent = False

emb_size = 512
action_size = 7
distribution_str = 'Categorical'

gamma = 0.99
train_env_num = 32
rollout_len = 32

ac_args = {'input_size': emb_size, 'action_size': action_size}
ppo_args = {
    'normalize_adv': True,
    'learning_rate': 3e-4, 'returns_estimator': 'v-trace',
    'ppo_n_epoch': 4, 'ppo_n_mini_batches': 4,
    'rollback_alpha': 0.0
}
train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 1000, 'rollout_len': rollout_len}

run_test_process = True
render_test_env = False
test_process_act_deterministic = False


def make_env():
    def make():
        return make_mario_env(world, level)
    return make


def make_ac_model(ac_device):
    def make_ac():
        return OneLayerActorCritic(**ac_args)
    model = AgentModel(
        ac_device, make_ac, distribution_str,
        make_obs_encoder=Encoder,
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
