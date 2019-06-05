# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:36:44 2017

@author: arnaud
"""
import tensorflow as tf
import numpy as np

def local_feature_maximizer(input, layer_id, construct_log):
    with tf.name_scope("local_feature_maximizer_"+layer_id):
        feature_flat = tf.reshape(construct_log["weight"][-1], [-1,int(input.get_shape()[-1])])
        feature_norm = tf.nn.l2_normalize(feature_flat,0)
        
        cos_features = tf.matmul(tf.transpose(feature_norm), feature_norm) #- np.identity(input.get_shape()[-1]
        cos_features = tf.nn.relu(cos_features)
        #tf.summary.scalar("cos_feature", tf.reduce_mean(cos_features))
        
        cos_features_m = tf.reduce_mean(-(cos_features-1.0),0)    
        
        tf.summary.scalar("cos_feature_m", tf.reduce_mean(cos_features_m))
        
        input_flat = tf.reshape(input, [-1,int(input.get_shape()[-1])])
        feature_usage = (tf.reduce_max(tf.abs(input_flat),0)-tf.reduce_min(tf.abs(input_flat),0))/(tf.reduce_max(tf.abs(input_flat),0)+0.0001) #(tf.abs(input_flat),0)tf.reduce_mean(tf.abs(input_flat),0)/(tf.reduce_max(tf.abs(input_flat),0)+0.0001)
        #input = input/tf.norm(feature_flat,axis=0)
        
        loss = -tf.reduce_mean(cos_features_m*feature_usage)
        
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        
        tf.summary.scalar("feature_usage", tf.reduce_mean(feature_usage))
        
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["scopes"][-1].name)
        if len(construct_log["scopes"])>1:    
            var_list+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["scopes"][-2].name)
        gradz = construct_log["optimizer"].compute_gradients(loss, var_list=var_list)
        construct_log["gradients"]+=gradz
        
        return input
        
def softmax_loss(input, layer_id, construct_log, labels):
    with tf.name_scope("softmax_layer_"+layer_id):
        if type(labels) is float:
            labels = tf.ones([tf.shape(input)[0]])*labels
        #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=input))
        th = tf.nn.sigmoid(input)
        cross_entropy = tf.reduce_mean(-labels*tf.log1p(th)-(1.0-labels)*tf.log1p(1.0-th))
        tf.summary.scalar("loss", cross_entropy)
        construct_log["losses"].append(cross_entropy)
        return input
        
def squared_error(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("squared_error_layer_"+layer_id):
        th = tf.nn.sigmoid(input)
        loss = tf.reduce_mean(tf.abs(th-target))*rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input
def squared_error_1(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("squared_error_1_layer_"+layer_id):
        loss = tf.reduce_mean(tf.square(input-target))*rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input
def raw_error(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("raw_error_layer_"+layer_id):
        loss = tf.reduce_mean(tf.reduce_max(tf.abs(input-target), axis=[1,2]))*0.0*rate+tf.reduce_mean(tf.abs(input-target))*rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input

def most_relevant_error(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("most_relevant_error_layer_" + layer_id):
        abs_error = tf.abs(input - target)
        most_relevant = tf.nn.relu(abs_error - tf.stop_gradient(tf.reduce_mean(abs_error)*0.6+tf.reduce_max(abs_error)*0.4))
        loss = tf.reduce_sum(most_relevant)
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return input

def maxmin_diff(input, layer_id, construct_log, rate=1.0):
    with tf.name_scope("maxmin_diff_"+layer_id):
        loss = tf.reduce_mean(tf.reduce_min(input, axis=[0,1,2])-tf.reduce_max(input, axis=[0,1,2]))*rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input
def l2_loss(input, layer_id, construct_log, scopes=["self"], rate=0.1):
    with tf.name_scope("l2_loss_layer_"+layer_id):
        loss_l = []
        for s in scopes:
            if s == "self":
                l_var=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["network_scope"].name)
            else:
                l_var=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=s.name)
            for v in l_var:
                loss_l.append(tf.nn.l2_loss(v))
        construct_log["losses"].append(sum(loss_l)*rate)
        return input
        
def dist_loss(input, layer_id, construct_log, num_entries=1000, rate=1.0):
    with tf.name_scope("dist_loss_"+layer_id):
        flatten = tf.reshape(input, [-1, input.get_shape().as_list()[-1]])
        selected = tf.gather(flatten, tf.random_uniform([num_entries], maxval=tf.shape(flatten)[0], dtype=tf.int32))
        loss = tf.reduce_mean(tf.nn.relu(tf.matmul(selected, tf.transpose(selected))))
        construct_log["losses"].append(loss*rate)
        tf.summary.scalar("loss", loss)
        return input
        
def compute_gradients(input, layer_id, construct_log, scopes=["self"], losses=[], clear_losses=False, add_regularization=True):
    with tf.name_scope("gradient_layer_"+layer_id):
        loss = sum(construct_log["losses"]+losses)
        tf.summary.scalar("loss_"+str(layer_id), loss)
        if add_regularization:
            loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*1e-5
        l_var = []
        for s in scopes:
            if s == "self":
                l_var+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["main_scope"].name)
            elif type(s) is str:
                l_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["network_scope"][s].name)
            else:
                l_var+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=s.name)
        gradz = construct_log["optimizer"].compute_gradients(loss, var_list=l_var)
        if "gradients" not in construct_log:
            construct_log["gradients"] = []
        construct_log["gradients"]+=gradz
        if clear_losses:
            construct_log["losses"]=[]
        return input
def trainer(input, layer_id,construct_log, external_gradz=[], global_step=True, apply_batchnorm=True):
    with tf.name_scope("trainer_"+layer_id):
        merged_gradz = []
        if "gradients" not in construct_log:
            construct_log["gradients"] = []
        gradients = construct_log["gradients"]+external_gradz
        for i in range(len(gradients)):
            if gradients[i][0] != None:
                lgw=[]
                for ig in range(i+1, len(gradients)):
                    if gradients[ig][1].name == gradients[i][1].name:
                        lgw.append(gradients[ig][0])
                if len(lgw)!=0:
                    merged_gradz.append((tf.reduce_mean(tf.stack(lgw), 0), gradients[i][1]))
                else:
                    merged_gradz.append(gradients[i])
        if global_step==True:
            if "global_step" not in construct_log:
                global_step = tf.get_variable("global_step", initializer=0)
                construct_log["global_step"] = global_step
            else:
                global_step = construct_log["global_step"]
        gradients=merged_gradz
        apply_gradients=[construct_log["optimizer"].apply_gradients(gradients, global_step=global_step)]

        training_ops = apply_gradients
        if apply_batchnorm:
            training_ops += tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(training_ops):
            return tf.identity(input)
def l2_norm_layer(input, layer_id, construct_log):
    return tf.nn.l2_normalize(input,-1)
def set_adam_optimizer(input, layer_id, construct_log, learning_rate=0.0001, epsilon=1e-8, beta=0.9):
    with tf.variable_scope("adamOptimizer_"+layer_id):
        construct_log["optimizer"]=tf.train.AdamOptimizer(learning_rate,epsilon=epsilon)
        return input
def initializer(input, layer_id, construct_log):
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    return input

def scalar_summary(input, layer_id, construct_log, name):
    with tf.variable_scope("scalar_summary_"+str(layer_id)):
        tf.summary.scalar(name, input)
        return input

def set_global_step(input, layer_id, construct_log):
    with tf.variable_scope("global_step_"+str(layer_id)):
        global_step = tf.get_variable("global_step", initializer=0)
        construct_log["global_step"] = global_step
    return input