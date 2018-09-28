
"""
core/__init__.py is a central component of NetBluePrint. It is used to gather every resources files availables.

It construct a dictionary of operations that can be used in a workflow's configuration.
It construct a dictionary of datasets that can be loaded in the workflow.

"""



import os, pkgutil
import importlib

operations = {}
awailable_datasets={}

import template_reader
from os import listdir
from os.path import isfile, join
import commentjson
import tensorflow as tf
import types
from dataset import dataset
import inspect

import builder
import printProgress

### PYTHON LAYER FILES SCANNING ###

root_dir=os.path.split(__file__)[0]+"/.."

operations_locations=["./operations/", root_dir+"/operations/"]
templates_locations=["./templates/", root_dir+ "/templates/"]
datasets_locations=["./datasets/", root_dir+"/datasets/"]
if "DATASET_PATH" in os.environ:
    datasets_locations.append(os.environ["DATASET_PATH"])

tmp_list=[]
for i in range(len(operations_locations)):
    operations_locations[i]=os.path.abspath(operations_locations[i])+"/"
    if os.path.exists(operations_locations[i]):
        tmp_list.append(operations_locations[i])
operations_locations=tmp_list

tmp_list=[]
for i in range(len(templates_locations)):
    templates_locations[i]=os.path.abspath(templates_locations[i])+"/"
    if os.path.exists(templates_locations[i]):
        tmp_list.append(templates_locations[i])
templates_locations=tmp_list
tmp_list=[]
for i in range(len(datasets_locations)):
    datasets_locations[i]=os.path.abspath(datasets_locations[i])+"/"
    if os.path.exists(datasets_locations[i]):
        tmp_list.append(datasets_locations[i])
datasets_locations=tmp_list

operations_locations=list(set(operations_locations))
templates_locations= list(set(templates_locations))
datasets_locations=list(set(datasets_locations))


modules = pkgutil.iter_modules(operations_locations)

for m in modules:
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

datasets_modules = pkgutil.iter_modules(datasets_locations)

for m in datasets_modules:
    #mod = m[0].#importlib.import_module(locations[1]+m)
    mod = m[0].find_module(m[1]).load_module(m[1])
    #print(dir(mod))
    contenant = dir(mod)
    for c in contenant:
        item = getattr(mod, c)
        if inspect.isclass(item) and issubclass(item, dataset) and item!= dataset:
            awailable_datasets[c]=item


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
        for k in l.keys():
            if isinstance(l[k], unicode):
                l[k] = l[k].encode('ascii','ignore')
        block_struct.append([type_, l])
    at = {}
    if "argument_translation" in blockConf:
        at = blockConf["argument_translation"]
    block_op = template_reader.create_block_operation(block_struct, name, at)
    if name in operations:
        raise Exception("Unavailable block name")
    else:
        operations[name] = block_op
        
### WRAPPING OF TENSORFLOW FUNCTIONS ###
        
def tf_function_wrapper(tf_function, name):
    def wrapped_function(input, layer_id, construct_log, **kw):
        with construct_log["printer"]("tensorflow "+name+" layer number " + str(layer_id)):
            with tf.variable_scope("TF_"+name+"_"+str(layer_id)):
                filtered_kw = {}
                for k in kw.keys():
                    if k in tf_function.__code__.co_varnames[1:tf_function.__code__.co_argcount]:
                        filtered_kw[k]=kw[k]
                    else:
                        if "printer" in construct_log and hasattr(construct_log["printer"], "printWarning"):
                            construct_log["printer"].printWarning(name+" has no argument \" " + k + " \", argument is ignored.")
                return tf_function(input, **filtered_kw)
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



builder.operations=operations
builder.awailable_datasets=awailable_datasets

create_workflow=builder.create_workflow
printProgress=printProgress.printProg

