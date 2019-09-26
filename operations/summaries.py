import tensorflow as tf

def scalar_summary(input, layer_id, construct_log, name):
    tf.summary.scalar(name, input)
    return input

def image_summary(input, layer_id, construct_log, name, max_outputs=16):
    tf.summary.image(name, input, max_outputs=max_outputs)
    return input


def merge_summaries(input, layer_id, construct_log):
    construct_log["summaries"] = tf.summary.merge_all()
    return input