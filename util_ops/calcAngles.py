# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 11:27:55 2017

@author: arnaud
"""

import tensorflow as tf
"""
    Calculate the angles between all the last dims of tensor
"""
def calcAngles(tensor):
    sizeFeatures = tensor.get_shape()[3].value
    tensor = tf.reshape(tensor, [tensor.get_shape()[0].value*tensor.get_shape()[1].value*tensor.get_shape()[2].value, sizeFeatures])
    print(tensor.get_shape())
    scalar_prod=tf.matmul(tf.transpose(tensor), tensor)
    norm = tf.sqrt(tf.reduce_sum(tf.square(tensor), 0, keep_dims=True))
    norm_mat = tf.matmul(tf.transpose(norm), norm)
    return tf.div(scalar_prod, norm_mat+0.0001)
    
def summarize_angles(tensor):
    tf.summary.scalar("angle_features", tf.reduce_mean(calcAngles(tensor)))