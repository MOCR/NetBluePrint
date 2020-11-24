from .path_dict import PathDict

def path_argument_translation(argument, construct_log, current_layer):
    if argument.startswith("@:/"):
        var_path = argument[3:]
        argument = construct_log[var_path]
    elif argument == "@input":
        argument = current_layer
    return argument

def translate_arguments(c, construct_log, current_layer, max_depth=2):
    if isinstance(c, list):
        indexs = range(len(c))
    elif isinstance(c, dict):
        indexs = list(c.keys())
    for i in indexs:
        if isinstance(c[i], list) or isinstance(c[i], dict) and max_depth > 0:
            translate_arguments(c[i], construct_log, current_layer, max_depth - 1)
        elif isinstance(c[i], str):
            c[i] = path_argument_translation(c[i], construct_log, current_layer)

def nbp_operation(function, opp_name=None):
    if opp_name is None:
        opp_name = function.__name__
    def decorated_function(input, construct_log, *args, **kwargs):
        if opp_name not in construct_log[construct_log["BUILD/local_path"] + "/layer_numbers"]:
            construct_log[construct_log["BUILD/local_path"] + "/layer_numbers/"+opp_name] = 0
        layer_id = str(construct_log[construct_log["BUILD/local_path"] + "/layer_numbers/"+opp_name])
        construct_log[construct_log["BUILD/local_path"] + "/layer_numbers/"+opp_name] += 1

        # construct_log["logger"].register_opp(function, opp_name)

        default_dict = construct_log[construct_log["BUILD/local_path"] + "/arguments"]

        if isinstance(default_dict, PathDict):
            default_dict_keys = list(default_dict.keys(leafs=True, recursive=True))
        else:
            default_dict_keys = list(default_dict.keys())

        args_behavior = "LEGACY"
        for v_name in default_dict_keys:
            split_v_name = v_name.split("/", 1)
            if len(split_v_name) > 1:
                if split_v_name[0] == opp_name or split_v_name[0] == opp_name + "_" + str(layer_id):
                    if args_behavior == "LEGACY" or split_v_name[1] not in kwargs:
                        kwargs[split_v_name[1]] = default_dict[v_name]
            else:
                add_var = False
                try:
                    if function.__code__.co_varnames.index(v_name) < function.__code__.co_argcount:
                        add_var = True
                except:
                    pass
                try:
                    if ("kw" in function.__code__.co_varnames and "args" in function.__code__.co_varnames) \
                            or (function.__code__.co_varnames.index("kw") <= function.__code__.co_argcount):
                        add_var = True
                except:
                    pass
                if add_var:
                    if v_name not in kwargs:
                        kwargs[v_name] = default_dict[v_name]

        translate_arguments(args, construct_log, input)
        translate_arguments(kwargs, construct_log, input)

        with construct_log["printer"](opp_name + "_" + layer_id):
            output = function(input, layer_id, construct_log, *args, **kwargs)
        return output