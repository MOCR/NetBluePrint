import tensorflow as tf
import os

def saver(input, layer_id, construct_log, path="./", restore=True,scope=None):
    with tf.name_scope("saver_" + layer_id):
        if "logger" in construct_log:
            restore = construct_log["restore"]
            path = construct_log["logger"].model_path
        if scope !=None:
            if type(scope) != str:
                scope=scope.name
            for v in tf.global_variables(scope=scope):
                print(v)
            s =tf.train.Saver(tf.global_variables(scope=scope))
        else:
            s = tf.train.Saver()
        sess = tf.get_default_session()
        #if not os.path.exists(path):
        #    os.makedirs(path)
        construct_log["printer"].addInfo(model_path=path)
        path = os.path.join(path, "model.ckpt")
        print((path+".index"))
        if restore and os.path.exists(path+".index"):
            s.restore(sess, path)

        construct_log["saver"] = lambda i : s.save(sess, path)
        return input

def load_checkpoint_scope(input, layer_id, construct_log, path, checkpoint_scope=None, scope_name=None):
    if scope_name is not None:
        scope_name = tf.get_variable_scope().name + "/"
    with construct_log["printer"]("loading " + scope_name + " variables from "+path):
        if checkpoint_scope == None:
            checkpoint_scope = scope_name
        assassignment_map = { checkpoint_scope : scope_name }
        tf.train.init_from_checkpoint(path, assassignment_map)
        return input
