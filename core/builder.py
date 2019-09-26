# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:40:00 2017

@author: Arnaud de Broissia
"""
import tensorflow as tf

import printProgress
import collections
import logger

operations = {}
awailable_datasets={}
awailable_filters={}

LAYER_COUNT=0
OP_COUNT=1

def path_argument_translation(argument, construct_log, current_layer):
    if argument.startswith("@:/"):
        var_path = argument.split("/")[1:]
        node = construct_log
        for vp in var_path:
            if type(node) is list:
                vp = int(vp)
            node = node[vp]
            argument = node
    elif argument == "@input":
        argument = current_layer
    return argument

def create_workflow(input,
                    configuration,
                    network_name,
                    reuse=None,
                    default_dict={},
                    printprog=printProgress.void_printer(),
                    parent_log=None,
                    construct_log_config={},
                    net_scope=None,
                    session_args={},
                    restore=False):
    """
    Create_workflow is the central function of NetBluePrint, it is used to build a workflow according to a specified configuration.

    :param input:
    :param configuration:
    :param network_name:
    :param reuse:
    :param default_dict:
    :param printprog:
    :param parent_log:
    :param construct_log_config:
    :return:
    """
    current_layer = input
    type_n = ""
    if parent_log == None:
        construct_log = {}
        construct_log["scopes"] = []
        construct_log["losses"] = []
        construct_log["weight"] = []
        construct_log["features"] = []
        construct_log["printer"] = printprog
        construct_log["default_dict"] = default_dict
        construct_log["awailable_datasets"]=awailable_datasets
        construct_log["awailable_filters"]=awailable_filters
        construct_log["awailable_operations"]=operations
        construct_log["reuse"] = reuse
        if type(restore) == int:
            run_to_restore=restore
            restore=True
        else:
            run_to_restore=-1
        construct_log["logger"] = logger.logger(name=network_name,
                                                construct_log=construct_log,
                                                struct=configuration,
                                                restore=restore,
                                                run_to_restore=run_to_restore)
        construct_log.update(construct_log_config)
        type_n = "Main"

        if tf.get_default_session() is None:
            default_session_args = {}
            default_session_args.update(session_args)
            sess = tf.Session(**default_session_args)
            construct_log["tf_session"] = sess
        else:
            sess = tf.get_default_session()
            construct_log["tf_session"] = sess

    else:
        construct_log = parent_log
        type_n = "Local"
    layer_numbers={}
    layer_numerotation = OP_COUNT

    def translate_arguments(c, construct_log, current_layer):
        for ic in range(len(c[1])):
            if type(c[1][ic]) is str:
                c[1][ic] = path_argument_translation(c[1][ic], construct_log, current_layer)
        for ic in c[2].keys():
            if type(c[2][ic]) is str:
                c[2][ic] = path_argument_translation(c[2][ic], construct_log, current_layer)

    with construct_log["tf_session"].as_default():
        with construct_log["printer"](type_n + " creation : " + network_name, timer=(type_n=="Main")):
            with tf.variable_scope(network_name if net_scope == None else net_scope, reuse=reuse) as scope:
                #print(scope.name)
                if type_n == "Main":
                    construct_log["main_scope"] = scope
                construct_log["local_scope"] = scope
                if type(configuration) is not list:
                    configuration=[configuration]
                i=0
                while i < len(configuration):
                    c = configuration[i]
                    if type(c) is not list:
                        c=[c]
                    c = list(c)
                    if len(c) < 2:
                        c.append([])
                    if len(c) < 3:
                        if type(c[1]) is list:
                            c.append({})
                        elif type(c[1]) is dict:
                            c.insert(1, [])
                        else:
                            print("Dont know...")
                    c[1] = list(c[1])
                    c[2] = dict(c[2])
                    opp = c[0]
                    if isinstance(opp, collections.Callable):
                        pass
                    elif opp in operations:
                        opp = operations[opp]
                    # enable setting current_layer from a variable in construct_log found with a specific path (starting with "@:/")
                    elif c[0].startswith("@:/"):
                        var_path = c[0].split("/")[1:]
                        node = construct_log
                        for vp in var_path:
                            if type(node) is list:
                                vp = int(vp)
                            node = node[vp]
                        current_layer = node
                        opp = None
                    # enable saving current_layer to a specific path in construct_log (starting with ">:/")
                    elif c[0].startswith(">:/"):
                        var_path = c[0].split("/")[1:]
                        node = construct_log
                        final = var_path[-1]
                        for vp in var_path[:-1]:
                            if vp not in node:
                                node[vp] = {}
                            node = node[vp]
                        if final.endswith(":[]"):
                            final = final.split(":")[0]
                            if final not in node:
                                node[final]=[]
                            node[final].append(current_layer)
                        else:
                            node[final] = current_layer
                        opp = None
                    elif c[0] == "if":
                        translate_arguments(c, construct_log, current_layer)
                        condition = c[2]["condition"] if "condition" in list(c[2].keys()) else c[1][0]
                        if condition:
                            additional_structure = c[2]["structure"] if "structure" in list(c[2].keys()) else c[1][-1]
                            if type(additional_structure) is not list:
                                additional_structure=[additional_structure]
                            configuration = configuration[:i+1]+additional_structure+configuration[i+1:]
                    else:
                        raise Exception("Unknown operation : " + c[0])
                    if opp != None:
                        if opp not in layer_numbers:
                            layer_numbers[opp]=0
                        if layer_numerotation == OP_COUNT:
                            layer_id = str(layer_numbers[opp])
                        else:
                            layer_id = str(i)
                        layer_numbers[opp]+=1
                        for v_name in list(default_dict.keys()):
                            split_v_name = v_name.split("/", 1)
                            if len(split_v_name) > 1:
                                if split_v_name[0] == c[0] or split_v_name[0] == c[0] + "_" + str(layer_id):
                                    c[2][split_v_name[1]] = default_dict[v_name]
                            else:
                                add_var = False
                                try:
                                    if opp.__code__.co_varnames.index(v_name) < opp.__code__.co_argcount:
                                        add_var = True
                                except:
                                    pass
                                try:
                                    if opp.__code__.co_varnames.index("kw") <= opp.__code__.co_argcount:
                                        add_var = True
                                except:
                                    pass
                                if add_var:
                                    if v_name not in c[2]:
                                        c[2][v_name] = default_dict[v_name]

                        translate_arguments(c, construct_log, current_layer)

                        construct_log["logger"].register_opp(opp)
                        current_layer = opp(current_layer, layer_id, construct_log, *c[1], **c[2])
                        construct_log["features"].append(current_layer)
                    construct_log["local_scope"] = scope
                    i+=1
    if type_n == "Main":
        construct_log["logger"].save_header()

    return current_layer, construct_log
