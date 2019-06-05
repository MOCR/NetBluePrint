
import tensorflow as tf

def log_value(input, layer_id, construct_log, name, value=None):
    with tf.name_scope("log_value_"+str(layer_id)):
        if value==None:
            value=input
        construct_log["logger"].register_value(name, value)
        return input

def log_finalize(input, layer_id, construct_log):
    with tf.name_scope("log_finalize_" + str(layer_id)):
        return construct_log["logger"].finalize_frame(input)
