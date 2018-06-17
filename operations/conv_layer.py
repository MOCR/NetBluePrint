# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:34:22 2017

@author: arnaud
"""

import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import batch_norm

def conv(input, layer_id,construct_log, out_size, kernel_size, reduce_factor=1, ortho=False):
    with tf.variable_scope("conv_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        if ortho:
            initializer = tf.orthogonal_initializer()
        else:
            initializer=tf.random_normal_initializer(stddev=0.02)
        w = tf.get_variable("weights", [kernel_size, kernel_size, input.get_shape()[-1], out_size], initializer=initializer, regularizer=tf.nn.l2_loss)
        output = tf.nn.conv2d(input,w, (1,reduce_factor,reduce_factor,1), "SAME")
        construct_log["weight"].append(w)
        b = tf.get_variable("bias", [out_size], initializer=tf.constant_initializer(0.0), regularizer=tf.nn.l2_loss)
        output+=b
        construct_log["scopes"].append(scope)
        return output
        
def dead_center_conv(input, layer_id,construct_log, out_size, kernel_size, reduce_factor=1, ortho=False):
    with tf.variable_scope("conv_dead_center_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        if ortho:
            initializer = tf.orthogonal_initializer()
        else:
            initializer=tf.random_normal_initializer(stddev=0.02)
        w = tf.get_variable("weights", [kernel_size, kernel_size, input.get_shape()[-1], out_size], initializer=initializer, regularizer=tf.nn.l2_loss)
        mask = np.ones([kernel_size, kernel_size, input.get_shape()[-1], out_size])
        mask[kernel_size/2][kernel_size/2]=mask[kernel_size/2][kernel_size/2]*0.0
        output = tf.nn.conv2d(input,w*mask, (1,reduce_factor,reduce_factor,1), "SAME")
        construct_log["weight"].append(w)
        b = tf.get_variable("bias", [out_size], initializer=tf.constant_initializer(0.0), regularizer=tf.nn.l2_loss)
        output+=b
        construct_log["scopes"].append(scope)
        return output

def conv1D(input, layer_id,construct_log, out_size, kernel_size, reduce_factor=1):
    with tf.variable_scope("conv1D_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        w = tf.get_variable("weights", [kernel_size, input.get_shape()[-1], out_size], initializer=tf.orthogonal_initializer(), regularizer=tf.nn.l2_loss)
        output = tf.nn.conv1d(input,w, reduce_factor, "SAME")
        construct_log["weight"].append(w)
        b = tf.get_variable("bias", [out_size], initializer=tf.constant_initializer(0.0), regularizer=tf.nn.l2_loss)
        output+=b
        #summarize_angles(w)
        construct_log["scopes"].append(scope)
        output=tf.nn.tanh(output)
    return output
    
def conv_expand(input, layer_id,construct_log, out_size, kernel_size, expand_Size):
    with tf.variable_scope("conv_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        raise Exception("Im using this?")
        w = tf.get_variable("weights", [kernel_size, kernel_size, input.get_shape()[-1], out_size*expand_Size*expand_Size], initializer=tf.contrib.layers.xavier_initializer_conv2d(), regularizer=tf.nn.l2_loss)
        construct_log["weight"].append(w)
        #b = tf.get_variable("bias", [out_size*expand_Size*expand_Size], initializer=tf.random_normal_initializer(), regularizer=tf.nn.l2_loss)
        output = tf.nn.conv2d(input,w, (1,1,1,1), "SAME")# + b
        shp = tf.shape(output)
        output = tf.reshape(output, [-1, shp[1]*expand_Size, shp[2]*expand_Size, out_size])
        #output = tf.nn.relu(output)
        construct_log["scopes"].append(scope)
    return output
def transpose_conv(input, layer_id,construct_log, out_size, kernel_size, expand_Size, preResize=True, ortho=False, grid_correction=False):
    with tf.variable_scope("transpose_conv_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        if preResize:
            input_Size = input.get_shape().as_list()
            #input = transpose_conv(input, 0,construct_log, input_Size[-1], expand_Size, expand_Size, preResize=False, ortho=True)
            resize = [int(input_Size[1])*expand_Size,int(input_Size[2])*expand_Size]
            input = tf.image.resize_bilinear(input, resize)
            # input += tf.random_normal(tf.shape(input), stddev=0.1)
            expand_Size = 1
        if ortho:
            initializer = tf.orthogonal_initializer()
        else:
            initializer=tf.random_normal_initializer(stddev=0.02)
        w = tf.get_variable("weights", [kernel_size, kernel_size,out_size, input.get_shape()[-1]], initializer=initializer, regularizer=tf.nn.l2_loss)       
        construct_log["weight"].append(w)
        b = tf.get_variable("bias", [out_size], initializer=tf.constant_initializer(0.0), regularizer=tf.nn.l2_loss)        
        input_Size = input.get_shape()#tf.shape(input) 
        #print input.get_shape()
        outsize = [tf.shape(input)[0], int(input_Size[1])*expand_Size,int(input_Size[2])*expand_Size, out_size]
        
        output = tf.nn.conv2d_transpose(input, w, outsize, [1, expand_Size, expand_Size, 1])
        if grid_correction:
            one_inp = tf.ones(tf.shape(input))
            grid_corrector = tf.nn.conv2d_transpose(one_inp, tf.ones(tf.shape(w)), outsize, [1, expand_Size, expand_Size, 1])
            grid_corrector = grid_corrector/tf.reduce_min(grid_corrector)
            #grid_corrector = tf.Print(grid_corrector, [grid_corrector])
            output = output/grid_corrector
        output = output+b
        
        #print output.get_shape()
        output = tf.reshape(output, outsize)
        construct_log["scopes"].append(scope)
    return output
    
def area_transpose_conv(input, layer_id,construct_log, out_size, kernel_size, expand_Size, preResize=True, ortho=False, grid_correction=False):
    with tf.variable_scope("transpose_conv_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        if preResize:
            input_Size = input.get_shape().as_list()
            #input = transpose_conv(input, 0,construct_log, input_Size[-1], expand_Size, expand_Size, preResize=False, ortho=True)
            resize = [int(input_Size[1])*expand_Size,int(input_Size[2])*expand_Size]
            input = tf.image.resize_bilinear(input, resize)
            # input += tf.random_normal(tf.shape(input), stddev=0.1)
            expand_Size = 1
        if ortho:
            initializer = tf.orthogonal_initializer()
        else:
            initializer=tf.random_normal_initializer(stddev=0.02)
        sh = tf.get_variable("share", [], initializer=tf.constant_initializer(0.5), regularizer=tf.nn.l2_loss)
        input=sh*conv(input, 66, construct_log, input.get_shape().as_list()[-1], 3)+(1.0-sh)*input
        w = tf.get_variable("weights", [kernel_size, kernel_size,out_size, input.get_shape()[-1]], initializer=initializer, regularizer=tf.nn.l2_loss)       
        construct_log["weight"].append(w)
        b = tf.get_variable("bias", [out_size], initializer=tf.constant_initializer(0.0), regularizer=tf.nn.l2_loss)        
        input_Size = input.get_shape()#tf.shape(input) 
        #print input.get_shape()
        outsize = [tf.shape(input)[0], int(input_Size[1])*expand_Size,int(input_Size[2])*expand_Size, out_size]
        
        output = tf.nn.conv2d_transpose(input, w, outsize, [1, expand_Size, expand_Size, 1])
        if grid_correction:
            one_inp = tf.ones(tf.shape(input))
            grid_corrector = tf.nn.conv2d_transpose(one_inp, tf.ones(tf.shape(w)), outsize, [1, expand_Size, expand_Size, 1])
            grid_corrector = grid_corrector/tf.reduce_min(grid_corrector)
            #grid_corrector = tf.Print(grid_corrector, [grid_corrector])
            output = output/grid_corrector
        output = output+b
        
        #print output.get_shape()
        output = tf.reshape(output, outsize)
        construct_log["scopes"].append(scope)
    return output
 
def transpose_conv1D(input, layer_id,construct_log, out_size, kernel_size, expand_Size, preResize=True, ortho=True):
    with tf.variable_scope("transpose_conv1D_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
         if len(input.get_shape().as_list()) != 4:
             input= tf.expand_dims(input,2)
         if preResize:
             input_Size = input.get_shape().as_list()
             #input = transpose_conv(input, 0,construct_log, input_Size[-1], expand_Size, expand_Size, preResize=False, ortho=True)
             resize = [int(input_Size[1])*expand_Size, 1]
             input = tf.image.resize_bilinear(input, resize)
             # input += tf.random_normal(tf.shape(input), stddev=0.1)
             expand_Size = 1
         if ortho:
             initializer = tf.orthogonal_initializer()
         else:
             initializer=tf.random_normal_initializer(stddev=0.02)
         w = tf.get_variable("weights", [kernel_size, 1,out_size, input.get_shape()[-1]], initializer=initializer, regularizer=tf.nn.l2_loss)       
         construct_log["weight"].append(w)
         b = tf.get_variable("bias", [out_size], initializer=tf.constant_initializer(0.0), regularizer=tf.nn.l2_loss)        
         input_Size = input.get_shape()#tf.shape(input) 
         #print input.get_shape()
         outsize = [tf.shape(input)[0], int(input_Size[1])*expand_Size, 1, out_size]
         output = tf.nn.conv2d_transpose(input, w, outsize, [1, expand_Size, 1, 1])+b
         #print output.get_shape()
         output = tf.reshape(output, outsize)
         output=tf.nn.tanh(output)
         construct_log["scopes"].append(scope)
         return output
         
def mean_var_summary(input, layer_id, construct_log, extra_name=""):
    with tf.variable_scope("mv_summay_"+str(layer_id), reuse=construct_log["reuse"]):
        mean_f = tf.reduce_mean(input, axis=-1)
        mean = tf.reduce_mean(mean_f)
        var = tf.reduce_mean(tf.square(mean_f-mean))
        tf.summary.scalar("mean_"+extra_name, mean)
        tf.summary.scalar("var_"+extra_name, var)
        return input
def tanh(input, layer_id, construct_log):
    with tf.variable_scope("tanh_"+str(layer_id), reuse=construct_log["reuse"]):
        return tf.nn.tanh(input)

def batch_norm_layer(input, layer_id, construct_log):
    with tf.variable_scope("batch_norm_" + str(layer_id), reuse=construct_log["reuse"])as batch_scope:
        out = batch_norm(input, is_training=True,
                           center=False, scale=True, updates_collections=None, scope=batch_scope, fused=True, epsilon=1e-5, decay = 0.9)
        return out

