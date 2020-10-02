
import math
import tensorflow as tf

from tensorflow.contrib.layers import batch_norm

SPLIT=0
SHUFFLE=1

def split_batchnorm(input, layer_id, construct_log, is_training=True, center=False, scale=True, renorm=True, do_split=True):
    if do_split:
        splited_input = tf.split(input, 2 )
        with tf.variable_scope("batch_norm_" + str(layer_id), reuse=construct_log["reuse"])as batch_scope:
            out_1 = batch_norm(splited_input[0], is_training=is_training,
                               center=center, scale=scale, scope=batch_scope, fused=True, epsilon=1e-5, decay = 0.999,  renorm=renorm)
        with tf.variable_scope("batch_norm_" + str(layer_id), reuse=True)as batch_scope:
            out_2 = batch_norm(splited_input[1], is_training=is_training,
                               center=center, scale=scale, scope=batch_scope, fused=True, epsilon=1e-5, decay = 0.999,  renorm=renorm)
        return tf.concat([out_1, out_2], axis=0)
    else:
        with tf.variable_scope("batch_norm_" + str(layer_id), reuse=construct_log["reuse"])as batch_scope:
            out = batch_norm(input, is_training=is_training,
                               center=center, scale=scale, scope=batch_scope, fused=True, epsilon=1e-5, decay = 0.999,  renorm=renorm)
            return out

def manifold_mix_initializer(input, layer_id, construct_log, labels, nb_mix_layers, proba_mix=0.5):
    with tf.variable_scope("manifold_mix_initializer_" + str(layer_id)):
        if proba_mix>1.0 or proba_mix<0.0:
            raise Exception("Invalide manifold mix proba, should be in range [0, 1], is "+str(proba_mix))
        construct_log["manifold_mix"]={}
        if type(labels) is not tf.Tensor:
            labels = tf.constant(labels)
        construct_log["manifold_mix"]["labels"]=labels
        if proba_mix==0:
            construct_log["manifold_mix"]["mix_control"]=-1
        else:
            construct_log["manifold_mix"]["mix_control"] = tf.random_uniform(shape=[], minval=0, maxval=int(math.ceil(nb_mix_layers/proba_mix)), dtype=tf.int32)
        return input

def manifold_mix_layer(input, layer_id, construct_log, num_mix_layer, mixing_strategy=SPLIT):
    with tf.variable_scope("manifold_mix_layer_" + str(layer_id)):
        batchsize = input.get_shape().as_list()[0]
        if batchsize == -1:
            batchsize = tf.shape(input)[0]
        def pass_by():
            return input, construct_log["manifold_mix"]["labels"]
        def mix_shuffle():
            shuffled_indexs = tf.random.shuffle(tf.range(tf.shape(input)[0]))
            shuffled_input = tf.gather(input, shuffled_indexs)
            shuffled_labels = tf.gather(construct_log["manifold_mix"]["labels"], shuffled_indexs)

            mix_shape = [batchsize]
            while len(mix_shape) < len(input.get_shape().as_list()):
                mix_shape.append(1)
            mix_parts = tf.random.uniform(mix_shape, minval=0.0, maxval=1.0)

            mixed_inputs = input*mix_parts + shuffled_input*(1.0-mix_parts)
            mixed_labels = construct_log["manifold_mix"]["labels"]*mix_parts + shuffled_labels * (1.0-mix_parts)
            return mixed_inputs, mixed_labels
        def mix_split():
            splited_input = tf.split(input, 2)
            splited_labels = tf.split(construct_log["manifold_mix"]["labels"], 2)

            mix_parts = tf.random.uniform([batchsize/2,1], minval=0.0, maxval=1.0)
            mix_parts_2 = tf.random.uniform([batchsize/2,1], minval=0.0, maxval=1.0)

            mix_parts_input = tf.reshape(mix_parts, [tf.shape(input)[0]/2,1,1,1])
            mix_parts_input_2 = tf.reshape(mix_parts_2, [tf.shape(input)[0]/2,1,1,1])
            mixed_inputs = splited_input[0] * mix_parts_input + splited_input[1] * (1.0-mix_parts_input)
            mixed_labels = splited_labels[0] * mix_parts + splited_labels[1] * (1.0-mix_parts)
            mixed_input_2 = splited_input[0] * mix_parts_input_2 + splited_input[1] * (1.0 - mix_parts_input_2)
            mixed_labels_2 = splited_labels[0] * mix_parts_2 + splited_labels[1] * (1.0 - mix_parts_2)
            return tf.concat([mixed_inputs,mixed_input_2], axis=0), tf.concat([mixed_labels, mixed_labels_2], axis=0)

        mix = mix_split if mixing_strategy==SPLIT else mix_shuffle

        out, labels = tf.cond(tf.math.equal(num_mix_layer,construct_log["manifold_mix"]["mix_control"]), mix, pass_by)
        construct_log["manifold_mix"]["labels"] = labels
        return out

