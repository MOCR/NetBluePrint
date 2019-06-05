

import tensorflow as tf

def set_placeholder(input, layer_id, construct_log, shape, name, dtype=tf.float32):
    if "placeholders" not in construct_log:
        construct_log["placeholders"]={}
    construct_log["placeholders"][name]=tf.placeholder(dtype=dtype, shape=shape, name=name)
    return input