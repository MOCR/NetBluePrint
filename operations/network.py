
import tensorflow as tf
from NetBluePrint.core import builder

def network(input, layer_id, construct_log, name, struct=None):
    if "networks" not in construct_log:
        construct_log["networks"]={}
    reuse=None
    if name in construct_log["networks"]:
        reuse=True
        struct=construct_log["networks"][name]
    else:
        if struct==None:
            raise Exception("Network "+name+" refered without structure or previous creation")
        construct_log["networks"][name]=struct
    net_output, _ = builder.create_network(input,struct,name,reuse=reuse,parent_log=construct_log)
    if reuse==None:
        if "network_scope" not in construct_log:
            construct_log["network_scope"]={}
        construct_log["network_scope"][name]=construct_log["local_scope"]
    return net_output