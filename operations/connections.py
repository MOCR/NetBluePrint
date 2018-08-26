import tensorflow as tf

def UStart(input, layer_id, construct_log):
    if "USkip" not in construct_log:
        construct_log["USkip"]=[]
    construct_log["USkip"].append(input)
    return input
    
def UEnd(input, layer_id, construct_log):
    input=tf.concat([input, construct_log["USkip"].pop()], axis=-1)
    return input
 
def input_layer(input, layer_id, construct_log, new_input):
    return new_input
def bridge_layer(input, layer_id, construct_log, bridge_name):
    if "bridges" not in construct_log:
        construct_log["bridges"]={}
    construct_log["bridges"][bridge_name]=input
    return input
def stop_grad(input, layer_id, construct_log):
    with tf.name_scope("stop_gradient"):    
        return tf.stop_gradient(input)

def concat(input, layer_id, construct_log, axis, item_to_concat):
    return tf.concat([input, item_to_concat], axis)
