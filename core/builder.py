# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:40:00 2017

@author: Arnaud de Broissia
"""
import tensorflow as tf

from .. import printProgress
import collections

operations = {}


def create_network(input, configuration, network_name, reuse=None, default_dict={},
                   printprog=printProgress.void_printer(), parent_log=None, construct_log_config={}):
    """

    :param input:
    :param configuration:
    :param network_name:
    :param reuse:
    :param optimizer:
    :return:
    """
    current_layer = input
    type_n = ""
    if parent_log == None:
        construct_log = {}
        construct_log["scopes"] = []
        construct_log["gradients"] = []
        construct_log["losses"] = []
        construct_log["weight"] = []
        construct_log["features"] = []
        construct_log["printer"] = printprog
        construct_log["default_dict"] = default_dict
        construct_log.update(construct_log_config)
        type_n = "Network"
    else:
        construct_log = parent_log
        type_n = "Block"
    with construct_log["printer"](type_n + " creation : " + network_name):
        with tf.variable_scope(network_name, reuse=reuse) as scope:
            if type_n == "Network":
                construct_log["network_scope"] = scope
            for i, c in enumerate(configuration):
                # print default_dict
                with tf.variable_scope("layer_" + str(i)):
                    # print c
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
                    else:
                        raise Exception("Unknown operation : " + c[0])
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
                    current_layer = opp(current_layer, str(i), construct_log, *c[1], **c[2])
                    construct_log["features"].append(current_layer)

    return current_layer, construct_log
