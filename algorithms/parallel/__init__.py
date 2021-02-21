import torch.multiprocessing as mp

from algorithms.parallel.model_process import ModelProcess
from algorithms.parallel.test_agent_process import TestAgentProcess
from algorithms.parallel.tb_writer_process import TensorBoardWriterProcess
from algorithms.parallel.train_agent import TrainAgent
from algorithms.parallel.optimizer_process import OptimizerProcess


def start_process(cls, *args):
    instance = cls(*args)
    process = mp.Process(target=instance.work)
    process.start()
    return process


__all__ = [
    'ModelProcess', 'TestAgentProcess',
    'TensorBoardWriterProcess', 'TrainAgent',
    'OptimizerProcess', 'start_process'
]
