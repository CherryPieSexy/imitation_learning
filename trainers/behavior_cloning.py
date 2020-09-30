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

        :param agent: BehaviorCloning instance
        :param demo_buffer:
        :param kwargs: test_env and log_dir
        """
        super().__init__(**kwargs)

        self._agent = agent
        self._demo_buffer = demo_buffer

    def save(self, filename):
        state_dict = {'agent': self._agent.state_dict()}
        torch.save(state_dict, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)
        self._agent.load_state_dict(checkpoint['agent'])

    def _train_step(self, batch, step):
        train_logs, time_logs, = self._agent.train_on_rollout(batch)
        self._write_logs('train/', train_logs, step)
        self._write_logs('time/', time_logs, step)

    def train(self, n_epoch, n_tests_per_epoch):
        self._save_n_test(0, n_tests_per_epoch, self._agent)
        train_step = 0

        for epoch in range(n_epoch):
            self._agent.train()
            for batch in tqdm(self._demo_buffer, ncols=90, desc=f'epoch_{epoch}'):
                self._train_step(batch, train_step)
                train_step += 1
            self._save_n_test(epoch + 1, n_tests_per_epoch, self._agent)
        self._writer.close()
