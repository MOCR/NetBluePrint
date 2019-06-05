import tensorflow as tf
import os

def saver(input, layer_id, construct_log, path="./model.ckpt", restore=True):
    with tf.name_scope("saver_" + layer_id):
        if "logger" in construct_log:
            restore = construct_log["logger"].restore
            path = construct_log["logger"].model_path
        s=tf.train.Saver()
        sess = tf.get_default_session()
        #if not os.path.exists(path):
        #    os.makedirs(path)
        print(path+".index")
        if restore and os.path.exists(path+".index"):
            s.restore(sess, path)

        construct_log["saver"] = lambda i : s.save(sess, path)
        return input