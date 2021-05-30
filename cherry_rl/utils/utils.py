import os
import time
import shutil


def create_log_dir(log_dir, file):
    # config saved in log_dir + 'config.py'
    # tensorboard logs saved in log_dir + 'tb/',
    # checkpoints in log_dir + 'checkpoints/'
    try:
        os.makedirs(log_dir)
        os.mkdir(log_dir + 'tb_logs')
        os.mkdir(log_dir + 'checkpoints')
    except FileExistsError:
        print('log_dir already exists')

    # save training config
    if os.path.exists(log_dir + 'config.py'):
        raise FileExistsError('config already exists')
    else:
        shutil.copy(file, log_dir + 'config.py')


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time
    return wrapper
