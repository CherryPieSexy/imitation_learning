from torch.utils.tensorboard import SummaryWriter


class TensorBoardWriterProcess:
    def __init__(self, log_dir, input_queue):
        self._log_dir = log_dir
        self._input_queue = input_queue

    def work(self):
        writer = SummaryWriter(self._log_dir + 'tb_logs/')
        try:
            while True:
                cmd, data = self._input_queue.get()
                if cmd == 'add_scalar':
                    tag, data_dict, step = data
                    for key, value in data_dict.items():
                        writer.add_scalar(tag + key, value, step)
                elif cmd == 'add_scalars':
                    # data = main_tag, data_dict, step
                    writer.add_scalars(*data)
                elif cmd == 'close':
                    break
                else:
                    raise NotImplementedError
                del data
        except KeyboardInterrupt:
            print('tb_writer worker: got KeyboardInterrupt')
        finally:
            writer.close()
