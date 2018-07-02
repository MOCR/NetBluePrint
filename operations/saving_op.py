import tensorflow as tf
import os

def saver(input, layer_id, construct_log, path="./model.ckpt", restore=True):
    with tf.name_scope("saver_" + layer_id):
        s=tf.train.Saver()
        sess = tf.get_default_session()
        #if not os.path.exists(path):
        #    os.makedirs(path)
        if restore and os.path.exists(path+".index"):
            #s.recover_last_checkpoints(path)
            #print(s._last_checkpoints)
            s.restore(sess, path)
        construct_log["saver"] = lambda i : s.save(sess, path)
        return input