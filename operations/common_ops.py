
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

def dropout(input, layer_id, construct_log, is_training=True, rate=0.5):
    if is_training:
        with tf.name_scope("dropout_"+str(layer_id)):
            construct_log["printer"].printResult("INFO", "Dropout set to training")
            return tf.nn.dropout(input,rate=rate)
    else:
        return input

def layer_noise(input, layer_id, construct_log, is_training):
    if is_training:
        with tf.name_scope("layer_noise_"+str(layer_id)):
            construct_log["printer"].printResult("INFO", "Layer noise set to training")
            noise = tf.random.normal(shape = input.get_shape().as_list(), stddev=0.01)
            return input+noise
    else:
        return input

def resize(input, layer_id, construct_log, size):
    with tf.name_scope("resize_" + str(layer_id)):
        return tf.image.resize_bilinear(input, size)

def random_blackout(input, layer_id, construct_log, rate=0.2):
    with tf.name_scope("random_blackout_" + str(layer_id)):
        # def pass_by():
        #     return input
        # def blackout():
        #     return tf.zeros(tf.shape(input))
        return tf.where(tf.greater(tf.random.uniform([tf.shape(input)[0]], minval=0.0, maxval=1.0), rate), input,  tf.zeros(tf.shape(input)))
def initiate_composite(input, layer_id, construct_log):
    construct_log["composition"]=tf.zeros([tf.shape(input)[0],256,256,3])
    return input

def l2_norm_layer(input, layer_id, construct_log):
    return tf.nn.l2_normalize(input,-1)

def compose(input, layer_id, construct_log):
    construct_log["composition"]+=input
    return input

def nb_channel(input, layer_id, construct_log):
    return input.get_shape().as_list()[-1]