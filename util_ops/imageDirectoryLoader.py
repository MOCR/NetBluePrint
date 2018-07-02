# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:38:56 2017

@author: arnaud
"""

import tensorflow as tf

from os import listdir
from os.path import isfile, join

# for first image loading for shape
import imageio
import numpy as np
import time


class imageDirectoryLoader:
    def __init__(self, batchsize, dirname, subdir=True, name="", coordinator=None):
        session = tf.get_default_session()
        print "BATCH SIZE = ", batchsize
        with tf.variable_scope("Image_Loader_" + name):
            with tf.device("/cpu:0"):
                if not subdir:
                    files = [dirname + f for f in listdir(dirname) if isfile(join(dirname, f))]
                    numImages = len(files)
                    files = tf.constant(files)

                else:
                    dirs = [dirname + f + "/" for f in listdir(dirname) if not isfile(join(dirname, f))]
                    images = {}
                    files = []
                    i = 0
                    labels = []
                    for d in dirs:
                        images[d] = [d + f for f in listdir(d) if isfile(join(d, f))]
                        files += images[d]
                        for _ in range(len(images[d])):
                            labels.append(i)
                        i += 1
                    numImages = len(labels)
                    if len(files) == 0:
                        raise Exception("Empty directory !")
                    files = tf.constant(files)
                    labels = tf.constant(labels)

                def genIndexs(size, maxConsec=4):
                    ret = []
                    while len(ret) < size:
                        r = np.random.randint(0, high=numImages)
                        c = np.random.randint(1, high=maxConsec)
                        for i in range(c):
                            if len(ret) == size or r + i >= numImages:
                                break
                            ret.append(r + i)
                    if len(ret) != size:
                        exit(-44)
                    return np.array(ret, dtype=np.int32)

                indexs = tf.py_func(genIndexs, [batchsize * 4], tf.int32)

                randomGather = tf.gather(files, indexs)

                op = lambda x: tf.image.decode_image(tf.read_file(x), channels=3)

                read = tf.map_fn(op, randomGather, dtype=tf.uint8)

                read = tf.to_float(read) / 256.0*2.0-1.0
                self.batch=read
                first_img = imageio.imread(session.run(files[0]))
                self.shape = first_img.shape

    def getBatch(self):
        return self.batch

    def getLabel(self):
        return 0  # self.labels

    def getShape(self):
        return self.shape
