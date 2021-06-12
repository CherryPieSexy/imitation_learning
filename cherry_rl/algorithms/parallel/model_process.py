from cherry_rl.algorithms.parallel.clone import clone


class ModelProcess:
    """
    Model should continuously run in process.
    Accepts all inputs via queue, send results back via pipes.
    """
    def __init__(
            self,
            model,  # model is created outside and shared between processes.
            input_queue,
            train_agent_pipe,
            test_agent_pipe
    ):
        self._model = model
        self._input_queue = input_queue
        self._train_agent_pipe = train_agent_pipe
        self._test_agent_pipe = test_agent_pipe

    def send_result(self, sender, result):
        result = clone(result)
        if sender == 'train_agent':
            self._train_agent_pipe.send(result)
        elif sender == 'test_agent':
            self._test_agent_pipe.send(result)
        else:
            raise ValueError(f'model got wrong sender {sender}')

    def work(self):
        try:
            while True:
                cmd, sender, data = self._input_queue.get()
                if cmd == 'act':
                    result = self._model.act(*data)
                elif cmd == 'reset_memory':
                    result = self._model.reset_memory_by_ids(*data)
                elif cmd == 'close':
                    break
                else:
                    raise NotImplementedError
                self.send_result(sender, result)
        except KeyboardInterrupt:
            print('model worker: got KeyboardInterrupt')
