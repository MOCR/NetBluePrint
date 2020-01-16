import tensorflow as tf
from tensorflow.contrib.staging import StagingArea as staging_area

def load_dataset(input, layer_id,construct_log, dataset, batchsize=-1, *args, **kwargs):
    with tf.variable_scope("dataset_" + layer_id):
        with tf.device("/cpu:0"):
            construct_log["dataset"]=construct_log["awailable_datasets"][dataset](construct_log, batchsize,*args, **kwargs)
            construct_log["data_inputs"]=construct_log["dataset"].get_datadict()
    return input

def bufferize_dataset(input, layer_id, construct_log):
    with tf.name_scope("dataset_queue_" + layer_id):
        with tf.device("/cpu:0"):
            batchsize = construct_log["dataset"].batchsize
            dtypes = []
            tensors = []
            names = []
            shapes = []
            for t in construct_log["data_inputs"].keys():
                tens = construct_log["data_inputs"][t]
                dtypes.append(tens.dtype)
                tensors.append(tens)
                names.append(t)
                shapes.append(tens.get_shape().as_list()[1:])
            fifo = tf.FIFOQueue(batchsize * 32, dtypes, shapes=shapes)
            batch = fifo.dequeue_many(batchsize)
            if type(batch) != list:
                batch=[batch]
            enqueue_op = fifo.enqueue_many(tensors)
            qr = tf.train.QueueRunner(fifo, [enqueue_op] * 6)
            coord = tf.train.Coordinator()
            sess = tf.get_default_session()

            threads = qr.create_threads(sess, coord=coord, daemon=True, start=True)
            construct_log["data_inputs"] = {}
            for i in range(len(batch)):
                construct_log["data_inputs"][names[i]] = batch[i]
            return input

def stage_dataset(input, layer_id,construct_log):
    with tf.name_scope("dataset_stager_" + layer_id):
        dtypes=[]
        tensors=[]
        names=[]
        shapes=[]
        for t in construct_log["data_inputs"].keys():
            tens = construct_log["data_inputs"][t]
            if type(tens) == tf.Tensor:
                dtypes.append(tens.dtype)
                tensors.append(tens)
                names.append(t)
                shapes.append(tens.get_shape().as_list())
        stager=staging_area(dtypes, capacity=1, shapes=shapes)
        construct_log["feed_stager:[]"]=stager.put(tensors)
        sess = tf.get_default_session()
        sess.run(construct_log["feed_stager"])
        staged_tensors=stager.get()
        #construct_log["data_inputs"] = {}
        for i in range(len(staged_tensors)):
            construct_log["data_inputs"][names[i]]=staged_tensors[i]
        return input

def feed_stager(input, layer_id,construct_log):
    with tf.name_scope("dataset_stage_feeder_" + layer_id):
        with tf.control_dependencies(construct_log["feed_stager"]):
            return tf.identity(input)