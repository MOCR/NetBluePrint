
import tensorflow as tf

def add_tensors(input, layer_id, construct_log, second_input):
    with tf.variable_scope("add_" + str(layer_id), reuse=construct_log["reuse"]):
        return input + second_input


def resnet_add(input, layer_id, construct_log, second_input):
    with tf.variable_scope("resnet_add_" + str(layer_id), reuse=construct_log["reuse"]):
        input_depth = input.get_shape().as_list()[-1]
        input_2_depth = second_input.get_shape().as_list()[-1]

        if input_depth != input_2_depth:
            if input_depth < input_2_depth:
                tmp = input
                input = second_input
                second_input = tmp

                tmp = input_depth
                input_depth = input_2_depth
                input_2_depth = tmp
            initializer = tf.random_normal_initializer(stddev=0.02)
            w = tf.get_variable("weights", [1, 1, input_2_depth, input_depth],
                                initializer=initializer, regularizer=tf.nn.l2_loss)
            second_input = tf.nn.conv2d(second_input, w, (1, 1, 1, 1), "SAME")

        return input + second_input