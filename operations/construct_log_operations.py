
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


def condition(input, layer_id, construct_log, condition, structure, else_structure=None):
    construct_log["printer"].printResult("INFO", "Evaluation IF structure")
    configuration_path = construct_log["BUILD/local_path"] + "/structure"
    configuration = construct_log[configuration_path]
    i = construct_log[construct_log["BUILD/local_path"] + "/build_cursor"]
    if condition:
        construct_log["printer"].printResult("INFO", "Condition is TRUE")

        if type(structure) is not list:
            structure = [structure]

        construct_log[configuration_path] = configuration[:i + 1] + structure + configuration[i + 1:]
    elif else_structure is not None:
        construct_log["printer"].printResult("INFO", "Condition is FALSE, alternative structure")
        if type(else_structure) is not list:
            else_structure = [else_structure]
        construct_log[configuration_path] = configuration[:i + 1] + else_structure + configuration[i + 1:]

    return input