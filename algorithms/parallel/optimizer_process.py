class OptimizerProcess:
    def __init__(self, optimizer, input_queue, queue_to_tb_writer):
        self._optimizer = optimizer
        self._input_queue = input_queue
        self._queue_to_tb_writer = queue_to_tb_writer
        self._steps_done = 0

    def clone(self, x):
        if x is None:
            return None
        elif type(x) is dict:  # rollout, observations
            return {k: self.clone(v) for k, v in x.items()}
        elif type(x) is tuple:  # actions
            return tuple([xx.clone() for xx in x])
        elif type(x) is bool:  # recurrent flag
            return x
        else:
            return x.clone()  # everything else

    def work(self):
        try:
            while True:
                cmd, data = self._input_queue.get()
                # qsize always 0, agent always trains on fresh data.
                # print(self._input_queue.qsize())
                if cmd == 'train':
                    data_clone = self.clone(data)
                    train_logs, time_logs = self._optimizer.train(data_clone)
                    self._steps_done += 1
                    self._queue_to_tb_writer.put(('add_scalar', ('train/', train_logs, self._steps_done)))
                    self._queue_to_tb_writer.put(('add_scalar', ('time/', time_logs, self._steps_done)))
                elif cmd == 'save':
                    self._optimizer.save(data)
                elif cmd == 'close':
                    break
                else:
                    raise NotImplementedError
        except KeyboardInterrupt:
            print('optimizer worker: got KeyboardInterrupt')
