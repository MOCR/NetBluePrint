
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

def CPU_server(input, layer_id, construct_log,name, struct=None, delet_losses_and_grad=True, **kwargs):
    with tf.device("/cpu:0"):
        if "losses" in construct_log:
            losses = list(construct_log["losses"])
        else:
            losses = []
        if "gradients" in construct_log:
            gradients = list(construct_log["gradients"])
        else:
            gradients = []

        kwargs["is_training"] = False
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        net_output = network(input,layer_id, construct_log, name, struct=struct, **kwargs)
        if delet_losses_and_grad:
            construct_log["losses"]=losses
            construct_log["gradients"] = gradients
            graph = tf.get_default_graph()
            graph.clear_collection(tf.GraphKeys.UPDATE_OPS)
            for opp in update_ops:
                graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, opp)
        return input


def all_GPU(input, layer_id, construct_log, name, struct=None, splits=[], **kwargs):
    pynvml.nvmlInit()
    nb_GPU = pynvml.nvmlDeviceGetCount()
    construct_log["number_of_GPUs"] = nb_GPU
    gpu_input = [None]*nb_GPU
    towers_args = []
    towers_dict = []
    kwargs["is_training"] = True
    for g in range(nb_GPU):
        towers_args.append(dict(kwargs))
        towers_dict.append(dict())

    original_data = {}
    for key in splits:
        if key == "input":
            gpu_input = tf.split(input, nb_GPU)
        if key.startswith("@:/"):
            value_to_split = construct_log[key[3:]]
            value_splits = tf.split(value_to_split, nb_GPU)
            for i, tdic in enumerate(towers_dict):
                tdic[key] = value_splits[i]
            original_data[key]=value_to_split

        elif key in kwargs.keys():
            if type(kwargs[key]) == str and kwargs[key].startswith("@:/"):
                value_to_split = construct_log[kwargs[key][3:]]
            else:
                value_to_split = kwargs[key]
            value_splits = tf.split(value_to_split, nb_GPU)
            for i, targs in enumerate(towers_args):
                targs[key]=value_splits[i]
    update_ops=None
    for i in range(nb_GPU):
        with tf.device("/gpu:"+str(i)):
            for key in towers_dict[i]:
                construct_log[key[3:]] = towers_dict[i][key]
            net_output = network(gpu_input[i], layer_id, construct_log, name, struct=struct, **towers_args[i])
            if update_ops is None:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    graph = tf.get_default_graph()
    graph.clear_collection(tf.GraphKeys.UPDATE_OPS)
    for opp in update_ops:
        graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, opp)

    for key in original_data:
        construct_log[key[3:]] = original_data[key]
    return input

def on_device(input, layer_id, construct_log, device, struct):
    with tf.device(device):
        net_output, _ = builder.create_workflow(input,
                                                struct,
                                                "Device_"+str(layer_id),
                                                parent_log=construct_log,
                                                scope_type="name")
    return net_output