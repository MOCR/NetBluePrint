
import tensorflow as tf

def tanh(input, layer_id, construct_log):
    with tf.variable_scope("tanh_"+str(layer_id), reuse=construct_log["reuse"]):
        return tf.nn.tanh(input)

def do_activation(input, layer_id, construct_log, activation_fn):
    if activation_fn is None:
        return input
    if type(activation_fn) == str or type(activation_fn) ==unicode:
        activation_fn = construct_log["awailable_operations"][activation_fn]
        return activation_fn(input, layer_id, construct_log)
    with tf.variable_scope(activation_fn.__name__ + "_"+str(layer_id), reuse=construct_log["reuse"]):
        return activation_fn(input)