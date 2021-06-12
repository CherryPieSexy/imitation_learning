import torch.multiprocessing as mp

from cherry_rl.algorithms.parallel.model_process import ModelProcess
from cherry_rl.algorithms.parallel.test_agent_process import TestAgentProcess
from cherry_rl.algorithms.parallel.tb_writer_process import TensorBoardWriterProcess
from cherry_rl.algorithms.parallel.train_agent import TrainAgent
from cherry_rl.algorithms.parallel.optimizer_process import OptimizerProcess


def start_process(cls, *args):
    instance = cls(*args)
    process = mp.Process(target=instance.work)
    process.start()
    return process


def run(
        log_dir,
        make_env,
        make_model,
        render_test_env,
        make_optimizer,
        train_agent_args, training_args,
        run_test_process=True,
        test_process_act_deterministic=True
):
    print(f'parallel experiment started in {log_dir}')
    model = make_model()
    # create communications
    queue_to_model = mp.Queue()
    model_to_train_agent, train_agent_from_model = mp.Pipe()
    model_to_test_agent, test_agent_from_model = mp.Pipe()

    queue_to_tester = mp.Queue()
    queue_to_writer = mp.Queue()
    queue_to_optimizer = mp.Queue()

    # start processes
    model_process = start_process(
        ModelProcess, model,
        queue_to_model, model_to_train_agent, model_to_test_agent
    )
    test_process = None
    if run_test_process:
        test_process = start_process(
            TestAgentProcess, make_env(), render_test_env,
            queue_to_tester, queue_to_model,
            test_agent_from_model, queue_to_writer,
            test_process_act_deterministic
        )
    tb_writer_process = start_process(
        TensorBoardWriterProcess, log_dir, queue_to_writer
    )
    optimizer_process = start_process(
        OptimizerProcess,
        model, make_optimizer,
        queue_to_optimizer, queue_to_writer
    )

    # train model
    train_agent = TrainAgent(
        make_env,
        **train_agent_args,
        queue_to_model=queue_to_model,
        queue_to_optimizer=queue_to_optimizer,
        pipe_from_model=train_agent_from_model,
        queue_to_tb_writer=queue_to_writer
    )
    try:
        train_agent.train(**training_args)
    except KeyboardInterrupt:
        print('main: got KeyboardInterrupt')
    finally:
        # kill processes
        train_agent.close()
        queue_to_optimizer.put(('save', log_dir + 'checkpoints/latest.pth'))
        queue_to_optimizer.put(('close', None))
        optimizer_process.join()
        queue_to_tester.put('close')
        if test_process is not None:
            test_process.join()
        queue_to_writer.put(('close', None))
        tb_writer_process.join()

        queue_to_model.put(('close', None, None))
        model_process.join()
        print('done!')
