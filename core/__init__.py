import os, pkgutil
import importlib

operations = {}

import template_reader
from os import listdir
from os.path import isfile, join
import commentjson
import tensorflow as tf
import types

### PYTHON LAYER FILES SCANNING ###

root_dir=os.path.split(__file__)[0]+"/.."
print(root_dir)

operations_locations=["./operations/", root_dir+"/operations/"]
templates_locations=["./templates/", root_dir+ "/templates/"]


modules = pkgutil.iter_modules(operations_locations)

for m in modules:
    print(m)
    #mod = m[0].#importlib.import_module(locations[1]+m)
    mod = m[0].find_module(m[1]).load_module(m[1])
    contenant = dir(mod)
    for c in contenant:
        item = getattr(mod, c)
        if callable(item) and hasattr(item, "__code__"):
            if item.__code__.co_argcount >= 3 and item.__code__.co_varnames[0] == "input" and item.__code__.co_varnames[1] == "layer_id" and item.__code__.co_varnames[2] == "construct_log":
                if item.func_name in operations and False:
                    raise Exception("Duplicate func_name in operations")
                else:
                    operations[item.func_name]=item

print(operations)

template_files=[]
for t_loc in templates_locations:
    template_files += [t_loc+ f for f in listdir(t_loc) if isfile(join(t_loc, f))]

### BLOCK FILES READING AND CONSTRUCTION OF BLOCK OPERATIONS ### 

for block in template_files:
    try:
        with open(block) as f:
            blockConf = commentjson.load(f)
    except Exception as e:
        print "ERROR: " +__file__+ " => Cannot read block file : " + block
        exit(-1)
    block_struct = []
    name = block.split("/")[-1].split(".")[0]
    for l in blockConf["structure"]:
        type_ = l["type"]
        del l["type"]
        block_struct.append([type_, l])
    at = {}
    if "argument_translation" in blockConf:
        at = blockConf["argument_translation"]
    block_op = block_creator.create_block_operation(block_struct, name, at)
    if name in operations:
        raise Exception("Unavailable block name")
    else:
        operations[name] = block_op
        
### WRAPPING OF TENSORFLOW FUNCTIONS ###
        
def tf_function_wrapper(tf_function, name):
    def wrapped_function(input, layer_id, construct_log, **kw):
        with construct_log["printer"]("tensorflow "+name+" layer number " + str(layer_id)):
            with tf.variable_scope("TF_"+name+"_"+str(layer_id)):
                return tf_function(input, **kw)
    return wrapped_function

for attr in dir(tf):
    obj = getattr(tf, attr)
    if isinstance(obj, types.FunctionType):
        if "tf."+attr in operations:
            raise Exception("TF wrapping error : Unavailable function name")
        else:
            operations["tf."+attr] = tf_function_wrapper(obj, attr)

for attr in dir(tf.contrib.layers):
    obj = getattr(tf.contrib.layers, attr)
    if isinstance(obj, types.FunctionType):
        if "tf.layers."+attr in operations:
            raise Exception("TF wrapping error : Unavailable function name")
        else:
            operations["tf.layers."+attr] = tf_function_wrapper(obj, "layers."+attr)
            
for attr in dir(tf.nn):
    obj = getattr(tf.nn, attr)
    if isinstance(obj, types.FunctionType):
        if "tf.nn."+attr in operations:
            raise Exception("TF wrapping error : Unavailable function name")
        else:
            operations["tf.nn."+attr] = tf_function_wrapper(obj, "nn."+attr)

print(operations.keys())

