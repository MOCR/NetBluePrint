import tensorflow as tf

def interpolation(input, layer_id, construct_log, expand_Size):
    with tf.name_scope("interpolation_" + str(layer_id)):
        input_Size = input.get_shape().as_list()
        resize = [int(input_Size[1]) * expand_Size, int(input_Size[2]) * expand_Size]
        input = tf.image.resize_bilinear(input, resize)
        return input

def apply_attention_mask(input, layer_id, construct_log, trunk, residual=True):
    with tf.name_scope("apply_attention_" + str(layer_id)):
        if residual:
            return (1.0+input)*trunk
        else:
            return input * trunk


