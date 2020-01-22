
import tensorflow as tf
from NetBluePrint.core import builder
import pynvml

def network(input, layer_id, construct_log, name, struct=None, var_scope=True, **kwargs):
    if "networks" not in construct_log:
        construct_log["networks"]={}
    pre_scope_reuse = construct_log["reuse"]
    reuse=None
    if name in construct_log["networks"]:
        reuse=True
        struct=construct_log["networks"][name]
        net_args=construct_log["network_default_params"][name]
        net_args.update(kwargs)
        net_scope=construct_log["network_scope"][name]
        if isinstance(net_scope, str):
            net_scope=None
    else:
        if struct==None:
            raise Exception("Network "+name+" refered without structure or previous creation")
        construct_log["networks"][name]=struct
        net_args=kwargs
        net_scope=None
    construct_log["reuse"] = reuse
    net_output, _ = builder.create_workflow(input,
                                            struct,
                                            name,
                                            reuse=reuse,
                                            parent_log=construct_log,
                                            default_dict=net_args,
                                            net_scope=net_scope,
                                            scope_type="VAR" if var_scope else "name")
    if reuse==None:
        if "network_scope" not in construct_log:
            construct_log["network_scope"]={}
        if "network_default_params" not in construct_log:
            construct_log["network_default_params"]={}
        construct_log["network_default_params"][name]=kwargs
        construct_log["network_scope"][name]=construct_log["local_scope"]
    construct_log["reuse"] = pre_scope_reuse
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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        losses_collection = tf.get_collection(tf.GraphKeys.LOSSES)
        regul = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        net_output = network(input,layer_id, construct_log, name, struct=struct, var_scope=False, **kwargs)
        if delet_losses_and_grad:
            construct_log["losses"]=losses
            construct_log["gradients"] = gradients
            graph = tf.get_default_graph()
            graph.clear_collection(tf.GraphKeys.UPDATE_OPS)
            for opp in update_ops:
                graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, opp)

            graph.clear_collection(tf.GraphKeys.SUMMARIES)
            for summ in summaries:
                graph.add_to_collection(tf.GraphKeys.SUMMARIES, summ)

            graph.clear_collection(tf.GraphKeys.LOSSES)
            for l in losses_collection:
                graph.add_to_collection(tf.GraphKeys.LOSSES, l)

            graph.clear_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            for r in regul:
                graph.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, r)
        return input


def all_GPU(input, layer_id, construct_log, name, struct=None, splits=[], **kwargs):
    with tf.device("/cpu:0"):
        pynvml.nvmlInit()
        nb_GPU = pynvml.nvmlDeviceGetCount()
        construct_log["number_of_GPUs"] = nb_GPU
        gpu_input = [None]*nb_GPU
        towers_args = []
        towers_dict = []
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
            net_output = network(gpu_input[i], layer_id, construct_log, name, struct=struct, var_scope=False, **towers_args[i])
            if update_ops is None:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    graph = tf.get_default_graph()
    graph.clear_collection(tf.GraphKeys.UPDATE_OPS)
    for opp in update_ops:
        graph.add_to_collection(tf.GraphKeys.UPDATE_OPS, opp)

    for key in original_data:
        construct_log[key[3:]] = original_data[key]
    return input


def nccl_GPU(input, layer_id, construct_log, name, struct=None, splits=[], **kwargs):
    from tensorflow.python.distribute import values as value_lib
    with tf.device("/cpu:0"):
        pynvml.nvmlInit()
        nb_GPU = pynvml.nvmlDeviceGetCount()
        construct_log["number_of_GPUs"] = nb_GPU
        gpu_input = [None]*nb_GPU
        towers_args = []
        towers_dict = []
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
    variables = []
    outs = []
    if "gradients" in construct_log:
        original_gradz = construct_log["gradients"]
    else:
        original_gradz = []
    destinations = []
    for i in range(nb_GPU):
        with tf.device("/gpu:"+str(i)):
            destinations.append("/gpu:"+str(i))
            for key in towers_dict[i]:
                construct_log[key[3:]] = towers_dict[i][key]

            replica_name = name if i == 0 else name+"_"+str(i)
            net_output = network(gpu_input[i],
                                 layer_id,
                                 construct_log,
                                 replica_name,
                                 struct=struct,
                                 var_scope=True,
                                 **towers_args[i])
            construct_log["gradients"] = original_gradz
            replica_variables = tf.global_variables(scope=construct_log["network_scope"][replica_name].name)
            replica_variables = sorted(replica_variables, key = lambda x : x.name)
            variables.append(replica_variables)
            outs.append(net_output)

    master = variables[0]

    for i, rep in enumerate(variables):
        with open("variables_"+str(i)+".csv", "w") as f:
            for var in rep:
                    f.write(var.name + "\n")

    variables = zip(*variables)

    nccl = tf.contrib.distribute.AllReduceCrossDeviceOps()

    with open("variables.csv", "w") as f:
        for var in variables:
            for v in var:
                f.write(v.name+ ", ")
            f.write("\n")

    for var in variables:
        print(var)
        print("\n")
        for replic in var[1:]:
            construct_log["initialization_opps:[]"]= tf.assign(replic, var[0])

    print(variables)
    synchronize = []
    for v in variables:
        print(v)
        print("\n")
        per_replica = value_lib.PerReplica({ device: var for device, var in zip(destinations, v)})
        synchronize += nccl.reduce(tf.distribute.ReduceOp.MEAN, per_replica, destinations)

    construct_log["printer"].printResult("INFO", "finished sync opps")

    with tf.control_dependencies(synchronize):
        with tf.control_dependencies(outs):
            construct_log["synchronize"] = tf.identity(0)

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

