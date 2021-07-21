import torch
import torch.multiprocessing as mp

from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.recurrent_encoders import OneLayerActorCritic
from cherry_rl.algorithms.nn.forward_dynamics_model import ForwardDynamicsDiscreteActionsModel
from cherry_rl.algorithms.nn.inverse_dynamics_model import InverseDynamicsModel
from cherry_rl.algorithms.nn.agent_model import AgentModel

from cherry_rl.algorithms.optimizers.rl.ppo import PPO
from cherry_rl.algorithms.optimizers.icm.forward_dynamics_optimizer import ForwardDynamicsModelOptimizer
from cherry_rl.algorithms.optimizers.bco.inverse_dynamics_optimizer import InverseDynamicsOptimizer
from cherry_rl.algorithms.optimizers.icm.icm import ICMOptimizer

import cherry_rl.algorithms.parallel as parallel

from configs.mario.encoder import Encoder
from configs.mario.env import make_mario_env


world, level = 1, 1
log_dir = f'logs/mario/world-{world}-level-{level}_icm/'
device = torch.device('cpu')
recurrent = False

ac_emb_size = 512
dynamics_emb_size = 512
action_size = 12
action_distribution_str = 'Categorical'
state_distribution_str = 'deterministic'

gamma = 0.99
train_env_num = 32
rollout_len = 64

ac_args = {'input_size': ac_emb_size, 'action_size': action_size}
ppo_args = {
    'entropy': 1e-2,
    'normalize_adv': True,
    'learning_rate': 3e-4, 'returns_estimator': 'v-trace',
    'ppo_n_epoch': 10, 'ppo_n_mini_batches': 8,
    'rollback_alpha': 0.0
}

fdm_args = {
    'observation_size': 512, 'hidden_size': 256, 'action_size': action_size,
    'action_distribution_str': action_distribution_str, 'state_distribution_str': state_distribution_str
}
fdm_optimizer_args = {'learning_rate': 3e-4, 'clip_grad': 0.5}

idm_args = {
    'observation_size': 512, 'hidden_size': 256, 'action_size': action_size,
    'distribution_str': action_distribution_str
}
idm_optimizer_args = {'learning_rate': 3e-4, 'clip_grad': 0.5}

icm_args = {}  # use default args

train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 100, 'rollout_len': rollout_len}

run_test_process = False
render_test_env = True
test_process_act_deterministic = False


def make_env():
    def make():
        return make_mario_env(world, level, complex_movement=True)
    return make


def make_ac_model(ac_device):
    def make_ac():
        return OneLayerActorCritic(**ac_args)
    model = AgentModel(
        ac_device, make_ac, action_distribution_str,
        make_obs_encoder=Encoder,
        value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    ppo = PPO(model, **ppo_args)

    fdm = ForwardDynamicsDiscreteActionsModel(**fdm_args)
    fdm_optimizer = ForwardDynamicsModelOptimizer(fdm, **fdm_optimizer_args)

    idm = InverseDynamicsModel(**idm_args)
    idm_optimizer = InverseDynamicsOptimizer(idm, **idm_optimizer_args)

    icm = ICMOptimizer(
        ppo, fdm_optimizer, idm_optimizer,
        dynamics_encoder_factory=Encoder,
        **icm_args
    )
    return icm


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
