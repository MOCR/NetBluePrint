import tensorflow as tf

def pixel_to_norm(input, layer_id,construct_log, train_normer=False, init_from_vault=True):
    with tf.variable_scope("pixel_to_norm_"+str(layer_id), reuse=construct_log["reuse"]) as scope:
        kernel_size = 1
        out_size=16
        if init_from_vault==True:
            initializer=construct_log["awailable_filters"]["pixel_to_norm.pkl"]()
            w = tf.get_variable("weights", initializer=initializer, regularizer=tf.nn.l2_loss, trainable=train_normer)
        else:
            initializer=tf.random_normal_initializer(stddev=0.02)
            w = tf.get_variable("weights", [kernel_size, kernel_size, input.get_shape()[-1], out_size], initializer=initializer, regularizer=tf.nn.l2_loss, trainable=train_normer)
        construct_log["pixel_to_norm_weights"]=w
        output = tf.nn.conv2d(input,w, (1,1,1,1), "SAME")

        output = tf.nn.l2_normalize(output, axis=-1)

        return output


def norm_to_pixel(input, layer_id, construct_log, train_normer=False, init_from_vault=True):
    with tf.variable_scope("norm_ti_pixel_" + str(layer_id), reuse=construct_log["reuse"]) as scope:
        kernel_size = 1
        out_size = 3
        input = tf.nn.l2_normalize(input, axis=-1)
        if init_from_vault==True:
            initializer=construct_log["awailable_filters"]["norm_to_pixel.pkl"]()
            w = tf.get_variable("weights", initializer=initializer, regularizer=tf.nn.l2_loss, trainable=train_normer)
        else:
            initializer = tf.random_normal_initializer(stddev=0.02)
            w = tf.get_variable("weights", [kernel_size, kernel_size, input.get_shape()[-1], out_size],
                            initializer=initializer, regularizer=tf.nn.l2_loss, trainable=train_normer)
        construct_log["norm_to_pixel_weights"] = w
        output = tf.nn.conv2d(input, w, (1, 1, 1, 1), "SAME")

        return output

