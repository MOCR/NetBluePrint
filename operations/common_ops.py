
import tensorflow as tf

from tensorflow.contrib.layers import batch_norm

def batch_norm_layer(input, layer_id, construct_log, is_training=True, center=False, scale=True, renorm=True):
    with tf.variable_scope("batch_norm_" + str(layer_id), reuse=construct_log["reuse"])as batch_scope:
        out = batch_norm(input, is_training=is_training,
                           center=center, scale=scale, scope=batch_scope, fused=True, epsilon=1e-5, decay = 0.999,  renorm=renorm)
        return out

def downsize_mean(input, layer_id, construct_log, reduce_factor):
    if reduce_factor==1:
        return input
    else:
        with tf.variable_scope("downsize_mean_" + str(layer_id), reuse=construct_log["reuse"]):
            return tf.nn.avg_pool(input, [1,reduce_factor,reduce_factor, 1], [1,reduce_factor,reduce_factor, 1], padding= "SAME")

def dropout(input, layer_id, construct_log, is_training=True):
    with tf.name_scope("dropout_"+str(layer_id)):
        return tf.nn.dropout(input,rate=0.5 if is_training else 0.0)
