# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:37:22 2017

@author: arnaud
"""

import tensorflow as tf

def gramMatrix_layer(input, layer_id,construct_log):
	with tf.name_scope("gram_matrix_"+str(layer_id)):
		features = tf.transpose(input, [0,3,1,2])
		input_shape = input.get_shape().as_list()
		features = tf.reshape(features, [-1,input_shape[3],input_shape[1]*input_shape[2]] )
		#features = tf.reduce_mean(features, 0)
		mat = tf.matmul(features, tf.transpose(features, [0,2,1]))
		print mat.get_shape()
		if "gram_matrix" not in construct_log:
			construct_log["gram_matrix"]=[]
		construct_log["gram_matrix"].append(mat)
		return input