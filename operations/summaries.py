import tensorflow as tf

def scalar_summary(input, layer_id, construct_log, name):
    tf.summary.scalar(name, input)
    return input

def mean_var_summary(input, layer_id, construct_log, extra_name=""):
    with tf.variable_scope("mv_summay_"+str(layer_id), reuse=construct_log["reuse"]):
        mean_f = tf.reduce_mean(input, axis=-1)
        mean = tf.reduce_mean(mean_f)
        var = tf.reduce_mean(tf.square(mean_f-mean))
        tf.summary.scalar("mean_"+extra_name, mean)
        tf.summary.scalar("var_"+extra_name, var)
        return input

def image_summary(input, layer_id, construct_log, name, max_outputs=16):
    tf.summary.image(name, input, max_outputs=max_outputs)
    return input


def merge_summaries(input, layer_id, construct_log):
    construct_log["summaries"] = tf.summary.merge_all()
    return input