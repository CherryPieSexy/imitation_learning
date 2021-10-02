def clone(x):
    if x is None:
        return None
    elif type(x) is dict:  # rollout, observations
        return {k: clone(v) for k, v in x.items()}
    elif type(x) is tuple:  # actions
        return tuple([clone(xx) for xx in x])
    elif type(x) is bool:  # recurrent flag
        return x
    else:
        return x.clone()  # everything else
