# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:36:44 2017

@author: arnaud
"""
import tensorflow as tf
import numpy as np

def compute_gradients(input, layer_id, construct_log, scopes=["self"], losses=[], clear_losses=False, add_regularization=True):
    with tf.name_scope("gradient_layer_"+layer_id):
        loss = sum(construct_log["losses"]+losses)
        tf.summary.scalar("loss_"+str(layer_id), loss)
        l_var = []
        regularization_collection = []
        for s in scopes:
            construct_log["printer"].printResult("INFO", "Using scope : " + str(s))
            if s == "self":
                scope = construct_log["main_scope"].name
            elif s.startswith("@:/"):
                scope = construct_log[s[3:]]
                if not isinstance(scope, str):
                    scope = scope.name
            elif type(s) is str:
                scope = construct_log["network_scope"][s].name
            else:
                scope = s.name
            l_var+=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            if add_regularization:
                regularization_collection += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=scope)
        if add_regularization:
            regularization_collection = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = sum(regularization_collection) / len(regularization_collection) * 1e-5
            tf.summary.scalar("regularization_loss_" + str(layer_id), regularization_loss)
            loss += regularization_loss

        # with open(scopes[0].replace(""))

        gradz = construct_log["optimizer"].compute_gradients(loss, var_list=l_var, colocate_gradients_with_ops=True)
        if "gradients" not in construct_log:
            construct_log["gradients"] = []
        construct_log["gradients"]+=gradz
        construct_log["total_losses:[]"]=loss
        if clear_losses:
            construct_log["losses"]=[]
        return input
def trainer(input, layer_id,construct_log, external_gradz=[], global_step=True, apply_batchnorm=True):
    with tf.name_scope("trainer_"+layer_id):
        merged_gradz = []
        if "gradients" not in construct_log:
            construct_log["gradients"] = []
        gradients = construct_log["gradients"]+external_gradz
        processed_var = []
        print()
        for gr in gradients:
            if "additive_margin" in gr[1].name:
                print(gr)
        print()
        for i in range(len(gradients)):
            if gradients[i][0] != None and gradients[i][1].name not in processed_var:
                lgw=[gradients[i][0]]
                for ig in range(i+1, len(gradients)):
                    if gradients[ig][1].name == gradients[i][1].name:
                        lgw.append(gradients[ig][0])
                if len(lgw)>1:
                    merged_gradz.append((tf.reduce_mean(tf.stack(lgw), 0), gradients[i][1]))
                else:
                    merged_gradz.append((lgw[0], gradients[i][1]))
                processed_var.append(gradients[ig][1].name)
        if global_step==True:
            if "global_step" not in construct_log:
                global_step = tf.get_variable("global_step", initializer=0)
                construct_log["global_step"] = global_step
            else:
                global_step = construct_log["global_step"]

        gradients=merged_gradz
        print()
        for gr in gradients:
            if "additive_margin" in gr[0].name:
                print(gr)
        print()
        apply_gradients=[construct_log["optimizer"].apply_gradients(gradients, global_step=global_step)]

        training_ops = apply_gradients
        if apply_batchnorm:
            batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            training_ops += batchnorm_ops
        with tf.control_dependencies(training_ops):
            return tf.identity(input)

def set_adam_optimizer(input, layer_id, construct_log, learning_rate=0.0001, epsilon=1e-8, beta=0.9):
    with tf.variable_scope("adamOptimizer_"+layer_id):
        construct_log["optimizer"]=tf.train.AdamOptimizer(learning_rate,epsilon=epsilon, beta1=beta )
        return input
def initializer(input, layer_id, construct_log):
    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    if "initialization_opps" in construct_log:
        sess.run(construct_log["initialization_opps"])
    return input

def set_global_step(input, layer_id, construct_log):
    with tf.variable_scope("global_step_"+str(layer_id)):
        global_step = tf.get_variable("global_step", initializer=0)
        construct_log["global_step"] = global_step
    return input

def apply_updates(input, layer_id, construct_log, scopes=["self"]):
    updates = []
    if type(scopes) != list:
        scopes = [scopes]
    for s in scopes:
        if s == "self":
            updates += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=construct_log["main_scope"].name)
        elif type(s) is str:
            updates += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=construct_log["network_scope"][s].name)
        else:
            updates += tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=s.name)
    with tf.name_scope("apply_updates"):
        with tf.control_dependencies(updates):
            return tf.identity(input)