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


def multi_head_attention(input, layer_id, construct_log, nb_head, attention_target=None):
    with tf.name_scope("multi_head_attention_"+str(layer_id)):
        shape = input.get_shape().as_list()
        for i in range(len(shape)):
            if shape[i]==None:
                shape[i]=-1
        # if cosinus:
        #     input = tf.nn.l2_normalize(input, axis=-1)

        def vector_decomposer(vectors):
            vectors = tf.split(input, 3, axis=-1)
            for i, vect in enumerate(vectors):
                vect_size = vect.get_shape().as_list()[-1]
                vect = tf.reshape(vect, [shape[0], shape[1], nb_head, int(vect_size/nb_head)])
                vect = tf.transpose(vect, [0,2,1,3])
                vectors[i] = vect
            query_vectors = vectors[0]
            key_vectors = vectors[1]
            value_vectors = vectors[2]
            return query_vectors, key_vectors, value_vectors

        query_vectors, key_vectors, value_vectors = vector_decomposer(input)

        if attention_target is not None:
            _ , key_vectors, value_vectors = vector_decomposer(attention_target)

        scores = tf.matmul(query_vectors, tf.transpose(key_vectors, [0,1,3,2]))
        if not cosinus:
            scores = tf.nn.softmax(scores)
        out = tf.matmul(scores, value_vectors)
        out = tf.transpose(out, [0, 2, 1, 3])
        out_shape = list(shape)
        out_shape[-1] = out_shape[-1] / 3
        out = tf.reshape(out, out_shape)
        return out



def fully_connected_norm(input, layer_id, construct_log, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope("fully_connected_norm_"+str(layer_id)):
        n_state = input.get_shape().as_list()[-1]
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(input, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(input-u), axis=axis, keepdims=True)
        input = (input - u) * tf.rsqrt(s + epsilon)
        input = input*g + b
        return input