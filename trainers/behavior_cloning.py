import torch
from tqdm import tqdm


from trainers.base_trainer import BaseTrainer


class BehaviorCloningTrainer(BaseTrainer):
    def __init__(
            self,
            agent,
            demo_buffer,
            **kwargs
    ):
        """
        Behavior cloning trainer.
        Iterates trough demo buffer and performs one update (train-op) of agent per batch,
        periodically (once per epoch) tests and saves agent.

        :param agent: BehaviorCloning instance
        :param demo_buffer:
        :param kwargs: test_env and log_dir
        """
        super().__init__(**kwargs)

        self._agent = agent
        self._demo_buffer = demo_buffer

    def _train_step(self, batch, step):
        train_logs, time_logs, = self._agent.train_on_rollout(batch)
        self._write_logs('train/', train_logs, step)
        self._write_logs('time/', time_logs, step)

    def train(self, n_epoch, n_tests_per_epoch):
        self._agent.save(self._log_dir + 'checkpoints/' + f'epoch_{0}.pth')
        self._test_agent(0, n_tests_per_epoch, self._agent)
        train_step = 0

        for epoch in range(n_epoch):
            self._agent.train()
            p_bar = tqdm(self._demo_buffer, ncols=90, desc=f'epoch_{epoch}')
            for batch in p_bar:
                self._train_step(batch, train_step)
                train_step += 1
            self._agent.save(self._log_dir + 'checkpoints/' + f'epoch_{epoch + 1}.pth')
            self._test_agent(epoch + 1, n_tests_per_epoch, self._agent)
        self._writer.close()
