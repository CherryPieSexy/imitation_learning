import numpy as np


def _select(select_from, row, col):
    if type(select_from) is dict:
        return {key: _select(value, row, col) for key, value in select_from.items()}
    elif type(select_from) is tuple:
        return tuple(value[row, col] for value in select_from)
    elif select_from is None:
        return None
    else:
        return select_from[row, col]


def _feed_forward_data_generator(data_dict, num_batches):
    time, batch = data_dict['mask'].size()
    num_transitions = time * batch
    batch_size = num_transitions // num_batches

    flatten_indices = np.arange(num_transitions)
    np.random.shuffle(flatten_indices)

    for _, start_id in enumerate(range(0, num_transitions, batch_size)):
        selected_indices = flatten_indices[start_id:start_id + batch_size]
        row = selected_indices // batch
        col = selected_indices - batch * row

        yield _select(data_dict, row, col)


def _select_col(select_from, col):
    if type(select_from) is dict:
        return {key: _select_col(value, col) for key, value in select_from.items()}
    elif type(select_from) is tuple:
        # memory should have batch as 2-nd dimension (i.e. index '1').
        # it means that this function works fine with memory.
        return tuple([value[:, col] for value in select_from])
    elif select_from is None:
        return None
    else:
        return select_from[:, col]


def _recurrent_data_generator(data_dict, num_sequences):
    batch = data_dict['mask'].size1
    step_size = batch // num_sequences
    batch_indices = np.arange(batch)
    np.random.shuffle(batch_indices)

    for _, time_id in enumerate(range(0, batch, step_size)):
        col = batch_indices[time_id:time_id + step_size]
        selected = _select_col(data_dict, col)
        selected['observations'] = selected['observations'][:-1]
        yield selected


def get_data_generator(data_dict, recurrent, num_batches):
    """
    Creates data generator for multiple optimization steps on one rollout.
    Data generator returns dict with selected data pieces.

    :param data_dict: dict with data to select from.
    :param recurrent: controls data generation strategy.
    :param num_batches: number of elements in one data generator.
    :return: data generator.
    """
    if recurrent:
        return _recurrent_data_generator(data_dict, num_batches)
    else:
        return _feed_forward_data_generator(data_dict, num_batches)
