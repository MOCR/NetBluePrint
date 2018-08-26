
import tensorflow as tf
from NetBluePrint.core import builder

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
    net_output, _ = builder.create_workflow(input, struct, name, reuse=reuse, parent_log=construct_log, default_dict=net_args, net_scope=net_scope)
    if reuse==None:
        if "network_scope" not in construct_log:
            construct_log["network_scope"]={}
        if "network_default_params" not in construct_log:
            construct_log["network_default_params"]={}
        construct_log["network_default_params"][name]=kwargs
        construct_log["network_scope"][name]=construct_log["local_scope"]
    return net_output