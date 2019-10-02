
import tensorflow as tf
from NetBluePrint.core import builder
import pynvml

def network(input, layer_id, construct_log, name, struct=None, **kwargs):
    if "networks" not in construct_log:
        construct_log["networks"]={}
    reuse=None
    if name in construct_log["networks"]:
        reuse=True
        struct=construct_log["networks"][name]
        net_args=construct_log["network_default_params"][name]
        net_args.update(kwargs)
        net_scope=construct_log["network_scope"][name]
    else:
        if struct==None:
            raise Exception("Network "+name+" refered without structure or previous creation")
        construct_log["networks"][name]=struct
        net_args=kwargs
        net_scope=None
    net_output, _ = builder.create_workflow(input,
                                            struct,
                                            name,
                                            reuse=reuse,
                                            parent_log=construct_log,
                                            default_dict=net_args,
                                            net_scope=net_scope,
                                            scope_type="VAR")
    if reuse==None:
        if "network_scope" not in construct_log:
            construct_log["network_scope"]={}
        if "network_default_params" not in construct_log:
            construct_log["network_default_params"]={}
        construct_log["network_default_params"][name]=kwargs
        construct_log["network_scope"][name]=construct_log["local_scope"]
    return net_output

def CPU_server(input, layer_id, construct_log, struct, delet_losses_and_grad=True, **kwargs):
    with tf.device("/cpu:0"):
        reuse = None
        if "losses" in construct_log:
            losses = list(construct_log["losses"])
        else:
            losses = []
        if "gradients" in construct_log:
            gradients = list(construct_log["gradients"])
        else:
            gradients = []
        net_output, _ = builder.create_workflow(input, struct, "cpu_server", reuse=reuse, parent_log=construct_log,
                                                default_dict=kwargs, net_scope=None)
        if delet_losses_and_grad:
            construct_log["losses"]=losses
            construct_log["gradients"] = gradients
        return input


def all_GPU(input, layer_id, construct_log, struct, reuse=True, splits=[], **kwargs):
    pynvml.nvmlInit()
    nb_GPU = pynvml.nvmlDeviceGetCount()
    construct_log["number_of_GPUs"] = nb_GPU
    gpu_input = [input]*nb_GPU
    towers_args= []
    for g in range(nb_GPU):
        towers_args.append(dict(kwargs))
    for key in splits:
        if key == "input":
            gpu_input = tf.split(input, nb_GPU)
        elif key in kwargs.keys():
            if type(kwargs[key]) == str and kwargs[key].startswith("@:/"):
                var_path = kwargs[key].split("/")[1:]
                node = construct_log
                for vp in var_path:
                    if type(node) is list:
                        vp = int(vp)
                    node = node[vp]
                value_to_split = node
            else:
                value_to_split = kwargs[key]
            value_splits = tf.split(value_to_split, nb_GPU)
            for i, targs in enumerate(towers_args):
                targs[key]=value_splits[i]

    for i in range(nb_GPU):
        with tf.device("/gpu:"+str(i)):
            net_output, _ = builder.create_workflow(gpu_input[i], struct, "gpu_tower", reuse=reuse, parent_log=construct_log,
                                                default_dict=towers_args[i], net_scope=None)
    return input