# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:40:00 2017

@author: Arnaud de Broissia
"""
import tensorflow as tf

import printProgress
import collections

operations = {}
awailable_datasets={}


def create_workflow(input, configuration, network_name, reuse=None, default_dict={},
                    printprog=printProgress.void_printer(), parent_log=None, construct_log_config={}, net_scope=None):
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
        construct_log["reuse"] = reuse
        construct_log.update(construct_log_config)
        type_n = "Main"
    else:
        construct_log = parent_log
        type_n = "Local"
    with construct_log["printer"](type_n + " creation : " + network_name):
        with tf.variable_scope(network_name if net_scope == None else net_scope, reuse=reuse) as scope:
            print(scope.name)
            if type_n == "Main":
                construct_log["main_scope"] = scope
            construct_log["local_scope"] = scope
            if type(configuration) is not list:
                configuration=[configuration]
            for i, c in enumerate(configuration):
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
                else:
                    raise Exception("Unknown operation : " + c[0])
                if opp != None:
                    for v_name in list(default_dict.keys()):
                        split_v_name = v_name.split("/", 1)
                        if len(split_v_name) > 1:
                            if split_v_name[0] == c[0] or split_v_name[0] == c[0] + "_" + str(i):
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
                    for ic in c[2].keys():
                        if type(c[2][ic]) is str:
                            if c[2][ic].startswith("@:/"):
                                var_path = c[2][ic].split("/")[1:]
                                node = construct_log
                                for vp in var_path:
                                    if type(node) is list:
                                        vp = int(vp)
                                    node = node[vp]
                                c[2][ic] = node
                            elif c[2][ic] == "@input":
                                c[2][ic] = current_layer
                    print(c)
                    current_layer = opp(current_layer, str(i), construct_log, *c[1], **c[2])
                    construct_log["features"].append(current_layer)
            construct_log["local_scope"] = scope

    return current_layer, construct_log
