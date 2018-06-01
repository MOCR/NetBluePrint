

def load_dataset(input, layer_id,construct_log, dataset, batchsize, *args):
    construct_log["dataset"]=construct_log["awailable_datasets"][dataset](batchsize,*args)
    return input