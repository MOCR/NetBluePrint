
def save_in_list(input, layer_id, construct_log, list_name):
    if list_name not in construct_log:
        construct_log[list_name]=[]
    construct_log[list_name].append(input)
    return input
def clear_field(input, layer_id, construct_log, field_name):
    del construct_log[field_name]
    return input
def no_op(input, layer_id, construct_log):
    return input

