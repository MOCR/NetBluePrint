# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:26:52 2017

@author: arnaud
"""

import tensorflow as tf

import cifar.load_cifar as cifar
from NetBluePrint.core.dataset import dataset
import numpy as np

class cifar10(dataset):
    def __init__(self, batchsize = 64, resize_dim=None):
        super(cifar10, self).__init__(batchsize)
        with tf.name_scope("cifar10"):
            def get_batch(size):
                x = cifar.nextBatch(size)
                x=x.astype(np.float32)/255.0
                return x
            #self.batchsize = tf.placeholder_with_default(batchsize, [], "batch_size")
            self.batch = tf.py_func(get_batch, [self.batchsize], tf.float32)
            self.x_dim = 32
            self.y_dim = 32
            if resize_dim != None:
                self.batch = tf.image.resize_bilinear(self.batch, resize_dim)
                self.x_dim = resize_dim[0]
                self.y_dim = resize_dim[1]

            self.batch=tf.reshape(self.batch, [-1, self.x_dim, self.y_dim, 3])

            self.data_dict["image"]=self.batch