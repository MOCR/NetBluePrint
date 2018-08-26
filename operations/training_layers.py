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
        print(construct_log["losses"])
        loss = sum(construct_log["losses"]+losses)
        if add_regularization:
            loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.001
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
def trainer(input, layer_id,construct_log, external_gradz=[], LARS=False, master_learning_rate=0.001, trust_coef=0.001):
    with tf.name_scope("trainer_"+layer_id):
        merged_gradz = []
        if "gradients" not in construct_log:
            construct_log["gradients"] = []
        gradients = construct_log["gradients"]+external_gradz
        for i in range(len(gradients)):
            if gradients[i][0] != None:
                lgw=[]
                for ig in range(len(gradients)):
                    if gradients[ig][1].name == gradients[i][1].name:
                        lgw.append(gradients[ig][0])
                merged_gradz.append((tf.reduce_mean(tf.stack(lgw), 0), gradients[i][1]))
        gradients=merged_gradz
        for g in gradients:
            if g[0] != None:
                pass #tf.summary.scalar("w_grad_ratio_"+g[1].name, tf.norm(g[1])/tf.norm(g[0]))
                #tf.summary.scalar("w_" + g[1].name, tf.norm(g[1]))
                #tf.summary.scalar("grad_" + g[1].name, tf.norm(g[0]))
        if not LARS:
            apply_gradients=[construct_log["optimizer"].apply_gradients(gradients)]
        else:
            apply_gradients=[]
            for g in gradients:
                if g[0] != None:
                    apply_gradients.append(tf.train.AdamOptimizer(learning_rate=master_learning_rate*trust_coef*tf.norm(g[1])/(tf.norm(g[0])+0.0001),name="local_adam_"+g[1].name.split(":")[0]).apply_gradients([g]))
        with tf.control_dependencies(apply_gradients):
            return tf.identity(input)
def l2_norm_layer(input, layer_id, construct_log):
    return tf.nn.l2_normalize(input,-1)
def set_adam_optimizer(input, layer_id, construct_log, learning_rate=0.0001, beta=0.9):
    with tf.variable_scope("adamOptimizer_"+layer_id):
        construct_log["optimizer"]=tf.train.AdamOptimizer(learning_rate, beta)
        return input
def initializer(input, layer_id, construct_log):
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    return input

def scalar_summary(input, layer_id, construct_log, name):
    with tf.variable_scope("scalar_summary_"+str(layer_id)):
        tf.summary.scalar(name, input)
        return input