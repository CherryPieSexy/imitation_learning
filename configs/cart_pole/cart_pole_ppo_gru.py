import gym
import torch
import torch.nn as nn
import torch.multiprocessing as mp

import cherry_rl.utils.env_wrappers as wrappers
from cherry_rl.utils.utils import create_log_dir

from cherry_rl.algorithms.nn.recurrent_encoders import init, CompositeRnnEncoder, OneLayerActorCritic
from cherry_rl.algorithms.nn.agent_model import AgentModel
from cherry_rl.algorithms.optimizers.ppo import PPO

import cherry_rl.algorithms.parallel as parallel


log_dir = 'logs/cart_pole/exp_4_ppo_gru/'
device = torch.device('cpu')
recurrent = True

distribution_str = 'Categorical'

observation_size = 4
hidden_size = 16
action_size = 2

ac_args = {'input_size': hidden_size, 'action_size': action_size}
ppo_args = {
    'learning_rate': 0.01, 'returns_estimator': '1-step',
    'ppo_n_epoch': 2, 'ppo_n_mini_batches': 4,
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
    def make_encoder():
        def make_embedding():
            gain = nn.init.calculate_gain('tanh')
            embedding = nn.Sequential(
                init(nn.Linear(observation_size, hidden_size), gain=gain), nn.Tanh(),
            )
            return embedding
        return CompositeRnnEncoder(make_embedding, hidden_size, hidden_size)

    def make_ac():
        return OneLayerActorCritic(**ac_args)
    model = AgentModel(
        device, make_ac, distribution_str,
        make_obs_encoder=make_encoder
    )
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
