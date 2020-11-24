# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:40:00 2017

@author: Arnaud de Broissia
"""
import tensorflow as tf

import printProgress
import collections
import copy

from . import logger
from .utils import construct_log_pointer

from .path_dict import PathDict

operations = {}
awailable_datasets={}
awailable_filters={}

LAYER_COUNT=0
OP_COUNT=1

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
                    restore=False,
                    scope_type="VAR"):
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
        construct_log = PathDict()
        construct_log["scopes"] = []
        construct_log["losses"] = []
        construct_log["printer"] = printprog
        construct_log["default_dict"] = default_dict
        construct_log["awailable_datasets"] = awailable_datasets
        construct_log["awailable_filters"] = awailable_filters
        construct_log["awailable_operations"] = operations
        construct_log["BUILD/VERSIONS/tensorflow"] = tf.__version__
        construct_log["reuse"] = reuse
        if type(restore) == int:
            run_to_restore=restore
            restore=True
        else:
            run_to_restore=-1
        construct_log["restore"] = restore

        construct_log["logger"] = logger.logger(name=network_name,
                                                restore=restore,
                                                run_to_restore=run_to_restore)

        construct_log["logger"].register_opp({"structure" : configuration,
                                              "arguments" : default_dict}, network_name, "MAIN")
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

    elif isinstance(parent_log, PathDict):
        construct_log = parent_log
        type_n = "Local"
    else:
        raise Exception("Bad type : parent_log should be of type PathDict but is " + type(parent_log))
    construct_log["local_arguments"] = default_dict

    def get_scope(net_scope):
        if scope_type=="VAR":
            return tf.variable_scope(network_name if net_scope == None else net_scope, reuse=reuse)
        else:
            return tf.name_scope(network_name if net_scope == None else net_scope)

    with construct_log["tf_session"].as_default():
        with construct_log["printer"](type_n + " creation : " + network_name, timer=(type_n=="Main")):
            with get_scope(net_scope) as scope:
                if net_scope is not None:
                    construct_log["printer"].printResult("INFO", net_scope if isinstance(net_scope, str) else net_scope.name)
                construct_log["printer"].printResult("INFO", scope if isinstance(scope, str) else scope.name)
                if "BUILD/main_scope" not in construct_log:
                    construct_log["main_scope"] = scope
                    construct_log["BUILD/main_scope"] = scope
                construct_log["local_scope"] = scope
                construct_log["BUILD/local_scope"] = scope
                if "BUILD/local_path" in construct_log:
                    local_path = construct_log["BUILD/local_path"] + "/" + network_name
                else:
                    local_path = "BUILD/build_spaces/" + network_name
                construct_log["BUILD/local_path"] = local_path

                if type(configuration) is not list:
                    configuration = [configuration]

                construct_log[local_path + "/scope"] = scope
                construct_log[local_path + "/arguments"] = default_dict
                construct_log[local_path + "/structure"] = configuration
                construct_log[local_path + "/layer_numbers"] = {}

                configuration = construct_log_pointer(local_path + "/structure", construct_log)

                i = construct_log["BUILD/local_path"]+"/build_cursor"
                construct_log[i] = 0
                while construct_log[i] < len(configuration()):
                    c = configuration()[construct_log[i]]
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
                    elif isinstance(opp, tf.Tensor):
                        current_layer = opp
                        opp = None
                    # enable setting current_layer from a variable in construct_log found with a specific path (starting with "@:/")
                    elif c[0].startswith("@:/"):
                        var_path = c[0][3:]
                        current_layer = construct_log[var_path]
                        opp = None
                    # enable saving current_layer to a specific path in construct_log (starting with ">:/")
                    elif c[0].startswith(">:/"):
                        var_path = c[0][3:]
                        construct_log[var_path] = current_layer
                        opp = None
                    elif c[0].startswith("&:"):
                        opp_target = c[0][2:]
                        if opp_target not in default_dict:
                            raise Exception("Unknown operation alias : " + opp_target)
                        opp = operations[default_dict[opp_target]]
                    else:
                        raise Exception("Unknown operation : " + c[0])
                    if opp != None:
                        current_layer = opp(current_layer, construct_log, *c[1], **c[2])
                    construct_log["local_scope"] = scope
                    construct_log["local_arguments"] = default_dict
                    construct_log[i]+=1
                construct_log["BUILD/local_path"] = "/".join(construct_log["BUILD/local_path"].split("/")[:-1])

    if type_n == "Main":
        construct_log["logger"].save_data()

    return current_layer, construct_log
