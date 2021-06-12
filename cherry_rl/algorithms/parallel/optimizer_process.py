from cherry_rl.algorithms.parallel.clone import clone


class OptimizerProcess:
    def __init__(
            self,
            model, make_optimizer,
            input_queue, queue_to_tb_writer
    ):
        self._model = model
        self._make_optimizer = make_optimizer
        self._input_queue = input_queue
        self._queue_to_tb_writer = queue_to_tb_writer
        self._steps_done = 0

    def work(self):
        optimizer = self._make_optimizer(self._model)
        try:
            while True:
                cmd, data = self._input_queue.get()
                if cmd == 'train':
                    data_clone = clone(data)
                    train_logs, time_logs = optimizer.train(data_clone)
                    self._steps_done += 1
                    self._queue_to_tb_writer.put(('add_scalar', ('train/', train_logs, self._steps_done)))
                    self._queue_to_tb_writer.put(('add_scalar', ('time/', time_logs, self._steps_done)))
                elif cmd == 'save':
                    optimizer.save(data)
                elif cmd == 'close':
                    break
                else:
                    raise NotImplementedError
        except KeyboardInterrupt:
            print('optimizer worker: got KeyboardInterrupt')
