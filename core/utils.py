def construct_log_pointer(key, construct_log):
    return lambda : construct_log[key]