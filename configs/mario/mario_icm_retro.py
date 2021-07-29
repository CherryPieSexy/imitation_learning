from functools import partial

import torch
import torch.multiprocessing as mp

import retro

from cherry_rl.utils.utils import create_log_dir
import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.image_wrapper import ImageEnvWrapper

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
from configs.mario.env_retro import MarioWrapper


log_dir = f'logs/mario/icm_0/'
device = torch.device('cpu')
recurrent = False

ac_emb_size = 512
dynamics_emb_size = 512
action_size = 9
reward_size = 2
action_distribution_str = 'Bernoulli'
state_distribution_str = 'deterministic'
layer_norm = False

gamma = 0.99
train_env_num = 128
rollout_len = 32

ac_args = {
    'input_size': ac_emb_size, 'action_size': action_size, 'critic_size': reward_size
}
ppo_args = {
    'entropy': 1e-4,
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

icm_args = {
    'extrinsic_reward_weight': 0.0,
    'warm_up_steps': 1,
}

train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 1000, 'rollout_len': rollout_len}

run_test_process = False
render_test_env = True
test_process_act_deterministic = False


def make_env():
    def make():
        env = retro.make('SuperMarioBros-Nes')
        env = MarioWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env, 4)
        env = ImageEnvWrapper(env, x_size=84, y_size=84)
        env = wrappers.FrameStackWrapper(env, 4)
        return env

    return make


def make_ac_model(ac_device):
    def make_ac():
        return OneLayerActorCritic(**ac_args)
    model = AgentModel(
        ac_device, make_ac, action_distribution_str,
        make_obs_encoder=Encoder,
        value_normalizer_size=reward_size
    )
    return model


def make_optimizer(model):
    def make_ac_optimizer():
        return PPO(model, **ppo_args)

    def make_fdm_optimizer():
        fdm = ForwardDynamicsDiscreteActionsModel(**fdm_args).to(device)
        return ForwardDynamicsModelOptimizer(fdm, **fdm_optimizer_args)

    def make_idm_optimizer():
        idm = InverseDynamicsModel(**idm_args).to(device)
        return InverseDynamicsOptimizer(idm, **idm_optimizer_args)

    def make_encoder():
        encoder = Encoder().to(device)
        return encoder

    icm = ICMOptimizer(
        make_ac_optimizer=make_ac_optimizer,
        make_forward_dynamics_optimizer=make_fdm_optimizer,
        make_inverse_dynamics_optimizer=make_idm_optimizer,
        dynamics_encoder_factory=partial(make_encoder, layer_norm=layer_norm),
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
