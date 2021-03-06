import tensorflow as tf
import numpy as np


def additive_margin(input, layer_id, construct_log, labels, num_class,m=0.3,s=30, weight=1.0, center_loss=False):
    with tf.variable_scope("additive_margin_"+str(layer_id), reuse=construct_log["reuse"]):
        embedding_size = input.get_shape().as_list()[-1]
        weights_init=tf.random.normal(shape=[num_class, embedding_size], stddev=0.01)
        weights = tf.get_variable("centroids", initializer=weights_init)
        weights_normed = tf.nn.l2_normalize(weights, axis = -1)
        construct_log["centroids"] = weights_normed
        weights_normed_transposed = tf.transpose(weights_normed)

        normed_input = tf.nn.l2_normalize(input, axis=-1)
        cos = tf.matmul(normed_input, weights_normed_transposed)
        construct_log["cos_matchings"]=cos
        one_hot_label = tf.one_hot(labels, num_class)

        margined = tf.where(tf.math.equal(one_hot_label, 1.0), cos-m, cos)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(one_hot_label, margined*s))
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss*weight)
        if center_loss:
            sampled_weights = tf.gather(weights_normed, indices=tf.random_uniform(shape=[2048],maxval=num_class, dtype=tf.int32))
            center_loss = tf.reduce_mean(tf.matmul(sampled_weights, tf.transpose(sampled_weights)))
            tf.summary.scalar("center_loss", center_loss)
            construct_log["losses"].append(center_loss * 10.0)
        return input


def sampled_additive_margin(input, layer_id, construct_log, labels, num_class,m=0.3,s=30, weight=1.0, number_of_samples=20000):
    with tf.variable_scope("sampled_additive_margin_"+str(layer_id), reuse=construct_log["reuse"]):
        if num_class<number_of_samples:
            raise Exception("More samples than number of classes, abord..")

        embedding_size = input.get_shape().as_list()[-1]

        kernel = np.array(np.random.normal(scale=0.1, size=[num_class, embedding_size]), dtype=np.float32)
        range_of_classes = np.arange(0,num_class)
        def gen_kernel_indexs(labels_set):
            negatives = np.setdiff1d(range_of_classes, labels_set)
            sampled_negatives = np.random.choice(negatives, size=number_of_samples-labels_set.shape[0], replace=True)
            indexs = np.array(np.concatenate([labels_set,sampled_negatives], axis=0), dtype=np.int32)
            return indexs
        def sample_kernel(indexs):
            return kernel[indexs]

        def update_kernel(gradients, indexs, learning_rate=0.001):
            kernel[indexs] -= gradients*learning_rate
            return np.zeros([], dtype=np.int32)


        labels_set, new_labels = tf.unique(labels)
        labels = new_labels
        kernel_indexs = tf.py_func(gen_kernel_indexs, [labels_set], tf.int32)

        sampled_kernel = tf.py_func(sample_kernel, [kernel_indexs], tf.float32)

        weights_normed = tf.transpose(tf.nn.l2_normalize(sampled_kernel, axis = -1))

        normed_input = tf.nn.l2_normalize(input, axis=-1)
        cos = tf.matmul(normed_input, weights_normed)
        construct_log["cos_matchings"]=cos
        one_hot_label = tf.one_hot(labels, number_of_samples)

        margined = tf.where(tf.math.equal(one_hot_label, 1.0), cos-m, cos)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(one_hot_label, margined*s))

        special_gradient = tf.gradients(loss, [sampled_kernel])[0]
        apply_special_gradient = tf.py_func(update_kernel, [special_gradient, kernel_indexs], np.int32)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_special_gradient)
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss*weight)
        return input


def softmax_cross_entropy_loss(input, layer_id, construct_log, labels, num_class):
    with tf.name_scope("softmax_layer_"+layer_id):
        # labels = tf.Print(labels, [labels])
        one_hot_label = tf.one_hot(labels, num_class)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_label, logits=input))
        tf.summary.scalar("loss", cross_entropy)
        construct_log["losses"].append(cross_entropy)
        return input

def embedding_norm_loss(input, layer_id, construct_log, weight=1.0):
    with tf.name_scope("embedding_norm_loss" + layer_id):
        norm = tf.norm(input, axis = -1)
        construct_log["feature_vector_norm"] = norm
        loss = tf.reduce_mean(tf.abs(norm - 1.0)) * weight
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input

def embedding_dist_loss(input, layer_id, construct_log, weight=1.0):
    with tf.name_scope("embedding_dist_loss" + layer_id):
        loss = tf.reduce_mean(tf.matmul(input, tf.transpose(input))) * weight
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input


def binary_classification_loss(input, layer_id, construct_log, labels):
    with tf.name_scope("softmax_layer_" + layer_id):
        if type(labels) is float:
            labels = tf.ones([tf.shape(input)[0]]) * labels
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=input))
        th = tf.nn.sigmoid(input)
        cross_entropy = tf.reduce_mean(-labels * tf.log1p(th) - (1.0 - labels) * tf.log1p(1.0 - th))
        tf.summary.scalar("loss", cross_entropy)
        construct_log["losses"].append(cross_entropy)
        return input


def L2_loss(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("squared_error_layer_" + layer_id):
        loss = tf.reduce_mean(tf.square(input - target)) * rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input


def L1_loss(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("squared_error_1_layer_" + layer_id):
        loss = tf.reduce_mean(tf.abs(input - target)) * rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        construct_log["sq_loss"]=loss
        return input


def l2_regularization_loss(input, layer_id, construct_log, scopes=["self"], rate=0.1):
    with tf.name_scope("l2_loss_layer_" + layer_id):
        loss_l = []
        for s in scopes:
            if s == "self":
                l_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["network_scope"].name)
            else:
                l_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=s.name)
            for v in l_var:
                loss_l.append(tf.nn.l2_loss(v))
        construct_log["losses"].append(sum(loss_l) * rate)
        return input
