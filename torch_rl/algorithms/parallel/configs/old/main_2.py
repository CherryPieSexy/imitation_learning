import gym
import torch
import torch.multiprocessing as mp

from torch_rl import utils as wrappers
from torch_rl.utils import SubprocVecEnv
from torch_rl.utils.utils import create_log_dir
from torch_rl.algorithms import ActorCriticTwoMLP
from torch_rl.algorithms import AgentModel
from torch_rl.algorithms.optimizers.ppo import PPO

import torch_rl.algorithms.parallel as parallel


log_dir = 'logs_py/parallel/test_26/'
recurrent = False
gamma = 0.99
train_env_num = 32
device = torch.device('cpu')
distribution_str = 'Beta'
actor_critic_args = {'input_size': 24, 'hidden_size': 32, 'action_size': 2 * 4}
ppo_args = {
    'normalize_adv': True,
    'returns_estimator': 'v-trace',
    'ppo_n_epoch': 5, 'ppo_n_mini_batches': 4,
    'ppo_epsilon': 0.1
}
train_args = {'n_epoch': 20, 'n_steps_per_epoch': 500, 'rollout_len': 16}


def make_env():
    def make():
        env = gym.make('BipedalWalker-v3')
        env = wrappers.ContinuousActionWrapper(env)
        env = wrappers.ActionRepeatAndRenderWrapper(env)
        return env
    return make


def make_vec_env():
    vec_env = SubprocVecEnv([make_env() for _ in range(train_env_num)])
    return vec_env


def make_model():
    def make_ac():
        return ActorCriticTwoMLP(**actor_critic_args)
    model = AgentModel(
        device, make_ac, distribution_str,
        obs_normalizer_size=24,
        reward_normalizer_size=1,
        value_normalizer_size=1
    )
    return model


def make_optimizer(model):
    return PPO(model, **ppo_args)


def main():
    create_log_dir(log_dir, __file__)
    # create model & optimizer
    model = make_model()
    # model.share_memory()
    # optimizer = PPO(model, **ppo_args)

    # create communications
    queue_to_model = mp.Queue()
    model_to_train_agent, train_agent_from_model = mp.Pipe()
    model_to_test_agent, test_agent_from_model = mp.Pipe()

    queue_to_tester = mp.Queue()
    queue_to_writer = mp.Queue()
    queue_to_optimizer = mp.Queue()

    # start processes
    model_process = parallel.start_process(
        parallel.ModelProcess, model,
        queue_to_model, model_to_train_agent, model_to_test_agent
    )
    test_process = parallel.start_process(
        parallel.TestAgentProcess, make_env(), queue_to_tester,
        queue_to_model, test_agent_from_model,
        queue_to_writer
    )
    tb_writer_process = parallel.start_process(
        parallel.TensorBoardWriterProcess, log_dir, queue_to_writer
    )
    optimizer_process = parallel.start_process(
        parallel.OptimizerProcess,
        model, make_optimizer,
        queue_to_optimizer, queue_to_writer
    )

    # train model
    train_agent = parallel.TrainAgent(
        make_vec_env, gamma,
        recurrent, log_dir,
        queue_to_model, queue_to_optimizer,
        train_agent_from_model, queue_to_writer,
        n_plot_agents=0
    )
    try:
        train_agent.train(**train_args)
    except KeyboardInterrupt:
        print('main: got KeyboardInterrupt')
    finally:
        # kill processes
        train_agent.close()
        queue_to_optimizer.put(('close', None))
        optimizer_process.join()
        queue_to_tester.put('close')
        test_process.join()
        queue_to_writer.put(('close', None))
        tb_writer_process.join()

        queue_to_model.put(('close', None, None))
        model_process.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
