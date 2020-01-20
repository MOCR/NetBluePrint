# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:57:22 2017

@author: arnaud
"""

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import math


    
def fully_connected(input, layer_id,construct_log, out, outshape=None, function=None, bn=True, init="FAN", bias=True):
    with tf.variable_scope("fully_connected_"+str(layer_id)) as scope:
        input_shape = input.get_shape().as_list()
        nb_dim = len(input_shape)
        if -1 in input_shape or None in input_shape:
            input_shape = tf.shape(input)
        if init=="ORTHO":
            initializer = tf.orthogonal_initializer()
        elif init=="SMALL":
            initializer=tf.random_normal_initializer(stddev = 0.001)#tf.random_normal_initializer(stddev=0.00002)
        elif init=="FAN":
            initializer=tf.random_uniform_initializer(-1/math.sqrt(input.get_shape().as_list()[-1]), 1/math.sqrt(input.get_shape().as_list()[-1]))
        if nb_dim > 2:
            input = tf.reshape(input, [reduce(lambda x, y: x*y, input_shape[:-1]), input_shape[-1]])
        w = tf.get_variable("weights", [input.get_shape()[-1], out], initializer=initializer, regularizer=tf.nn.l2_loss)
        construct_log["weight"].append(w)
        output = tf.matmul(input, w)
        if bias:
            b = tf.get_variable("bias", [out], initializer=tf.zeros_initializer(), regularizer=tf.nn.l2_loss)
            output += b
        if function == "relu":
            output=tf.nn.relu(output)
        if function == "tanh":
            output=tf.nn.tanh(output)#+0.001*output
        if outshape!=None:
            output=tf.reshape(output, outshape)
            output= batch_norm(output, is_training=True,  
                           center=False, scale=True, updates_collections=None, scope=scope, fused=True, epsilon=1e-5, decay = 0.9)
        construct_log["scopes"].append(scope)
        if nb_dim > 2 :
            output = tf.reshape(output, input_shape[:-1] + [out])
    return output

def flatten(input, layer_id,construct_log):
    shape = input.get_shape().as_list()
    if shape[0] == None:
        shape[0] = -1
    size = 1
    for dim in shape[1:]:
        size *=dim
    return tf.reshape(input, [shape[0], size])