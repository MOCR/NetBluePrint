import tensorflow as tf
import numpy as np


def additive_margin(input, layer_id, construct_log, labels, num_class,m=0.3,s=30, weight=1.0, center_loss=False):
    with tf.variable_scope("additive_margin_"+str(layer_id)):
        embedding_size = input.get_shape().as_list()[-1]
        weights_init=tf.random.normal(shape=[num_class, embedding_size], stddev=0.01)
        weights = tf.get_variable("centroids", initializer=weights_init)
        weights_normed = tf.nn.l2_normalize(weights, axis = -1)
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
    with tf.variable_scope("additive_margin_"+str(layer_id)):
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


def local_feature_maximizer(input, layer_id, construct_log):
    with tf.name_scope("local_feature_maximizer_" + layer_id):
        feature_flat = tf.reshape(construct_log["weight"][-1], [-1, int(input.get_shape()[-1])])
        feature_norm = tf.nn.l2_normalize(feature_flat, 0)

        cos_features = tf.matmul(tf.transpose(feature_norm), feature_norm)  # - np.identity(input.get_shape()[-1]
        cos_features = tf.nn.relu(cos_features)
        # tf.summary.scalar("cos_feature", tf.reduce_mean(cos_features))

        cos_features_m = tf.reduce_mean(-(cos_features - 1.0), 0)

        tf.summary.scalar("cos_feature_m", tf.reduce_mean(cos_features_m))

        input_flat = tf.reshape(input, [-1, int(input.get_shape()[-1])])
        feature_usage = (tf.reduce_max(tf.abs(input_flat), 0) - tf.reduce_min(tf.abs(input_flat), 0)) / (
                    tf.reduce_max(tf.abs(input_flat),
                                  0) + 0.0001)  # (tf.abs(input_flat),0)tf.reduce_mean(tf.abs(input_flat),0)/(tf.reduce_max(tf.abs(input_flat),0)+0.0001)
        # input = input/tf.norm(feature_flat,axis=0)

        loss = -tf.reduce_mean(cos_features_m * feature_usage)

        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)

        tf.summary.scalar("feature_usage", tf.reduce_mean(feature_usage))

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["scopes"][-1].name)
        if len(construct_log["scopes"]) > 1:
            var_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=construct_log["scopes"][-2].name)
        gradz = construct_log["optimizer"].compute_gradients(loss, var_list=var_list)
        construct_log["gradients"] += gradz

        return input


def softmax_loss(input, layer_id, construct_log, labels):
    with tf.name_scope("softmax_layer_" + layer_id):
        if type(labels) is float:
            labels = tf.ones([tf.shape(input)[0]]) * labels
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=input))
        th = tf.nn.sigmoid(input)
        cross_entropy = tf.reduce_mean(-labels * tf.log1p(th) - (1.0 - labels) * tf.log1p(1.0 - th))
        tf.summary.scalar("loss", cross_entropy)
        construct_log["losses"].append(cross_entropy)
        return input


def squared_error(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("squared_error_layer_" + layer_id):
        th = tf.nn.sigmoid(input)
        loss = tf.reduce_mean(tf.abs(th - target)) * rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input


def squared_error_1(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("squared_error_1_layer_" + layer_id):
        loss = tf.reduce_mean(tf.abs(input - target)) * rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        construct_log["sq_loss"]=loss
        return input


def raw_error(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("raw_error_layer_" + layer_id):
        loss = tf.reduce_mean(tf.reduce_max(tf.abs(input - target), axis=[1, 2])) * 0.0 * rate + tf.reduce_mean(
            tf.abs(input - target)) * rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input


def most_relevant_error(input, layer_id, construct_log, target, rate=1.0):
    with tf.name_scope("most_relevant_error_layer_" + layer_id):
        abs_error = tf.abs(input - target)
        most_relevant = tf.nn.relu(
            abs_error - tf.stop_gradient(tf.reduce_mean(abs_error) * 0.6 + tf.reduce_max(abs_error) * 0.4))
        loss = tf.reduce_sum(most_relevant)
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
        return input


def maxmin_diff(input, layer_id, construct_log, rate=1.0):
    with tf.name_scope("maxmin_diff_" + layer_id):
        loss = tf.reduce_mean(tf.reduce_min(input, axis=[0, 1, 2]) - tf.reduce_max(input, axis=[0, 1, 2])) * rate
        tf.summary.scalar("loss", loss)
        construct_log["losses"].append(loss)
        return input


def l2_loss(input, layer_id, construct_log, scopes=["self"], rate=0.1):
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


def dist_loss(input, layer_id, construct_log, num_entries=1000, rate=1.0):
    with tf.name_scope("dist_loss_" + layer_id):
        flatten = tf.reshape(input, [-1, input.get_shape().as_list()[-1]])
        selected = tf.gather(flatten, tf.random_uniform([num_entries], maxval=tf.shape(flatten)[0], dtype=tf.int32))
        loss = tf.reduce_mean(tf.nn.relu(tf.matmul(selected, tf.transpose(selected))))
        construct_log["losses"].append(loss * rate)
        tf.summary.scalar("loss", loss)
        return input
