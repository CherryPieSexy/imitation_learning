import gym
import torch
import torch.multiprocessing as mp

import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.image_wrapper import ImageEnvWrapper
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.demo_buffer import TransitionsDemoBuffer
from cherry_rl.algorithms.nn.actor_critic import MLP, ActorCriticTwoMLP
from cherry_rl.algorithms.distributions import Beta
from cherry_rl.algorithms.nn.conv_encoders import DeepConvEncoder
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.rl.ppo import PPO
from cherry_rl.algorithms.optimizers.gail.discriminator_optimizer import DiscriminatorOptimizer, neg_log_sigmoid
from cherry_rl.algorithms.optimizers.gail.gail import GAIL

import cherry_rl.algorithms.parallel as parallel


log_dir = 'logs/car_racing/exp_4_gail_3/'
device = torch.device('cpu')
recurrent = False

obs_size = 256
hidden_size = 100
action_size = 3
distribution = Beta

gamma = 0.99
train_env_num = 8
rollout_len = 128

demo_file = 'demo_files/car_racing_20_ep.pickle'

ac_args = {
    'input_size': obs_size, 'hidden_size': hidden_size, 'action_size': 2 * action_size,
    'n_layers': 2, 'detach_actor': True, 'detach_critic': True
}
discriminator_args = {
    'input_size': obs_size, 'hidden_size': hidden_size, 'output_size': 1,
    'n_layers': 3, 'activation_str': 'tanh', 'output_gain': 1.0
}
ppo_args = {
    'normalize_adv': True,
    'returns_estimator': 'v-trace',
    'ppo_n_epoch': 3, 'ppo_n_mini_batches': 8,
    'rollback_alpha': 0.0
}
train_args = {
    'train_env_num': train_env_num, 'gamma': gamma, 'recurrent': recurrent,
    'log_dir': log_dir, 'n_plot_agents': 0
}
training_args = {'n_epoch': 10, 'n_steps_per_epoch': 250, 'rollout_len': rollout_len}

run_test_process = True
render_test_env = True
test_process_act_deterministic = True


def make_env():
    def make():
        env = gym.make('CarRacing-v0', verbose=0)
        env = wrappers.ContinuousActionWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env, 2)
        env = ImageEnvWrapper(env, x_size=96, y_size=96)
        env = wrappers.FrameStackWrapper(env, 4)
        return env
    return make


def make_ac_model(ac_device):
    def make_ac():
        return ActorCriticTwoMLP(**ac_args)
    model = AgentModel(
        ac_device, make_ac, distribution,
        make_obs_encoder=DeepConvEncoder,
        reward_scaler_size=1,
        value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    ppo = PPO(model, **ppo_args)
    discriminator = MLP(**discriminator_args)
    discriminator_optimizer = DiscriminatorOptimizer(
        discriminator, learning_rate=3e-4, clip_grad=0.5, reward_fn=neg_log_sigmoid
    )
    demo_buffer = TransitionsDemoBuffer(demo_file, train_env_num * rollout_len)
    gail = GAIL(ppo, discriminator_optimizer, demo_buffer, use_discriminator_grad=True)
    return gail


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
