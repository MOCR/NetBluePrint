# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:38:56 2017

@author: arnaud
"""

import tensorflow as tf

from os import listdir
from os.path import isfile, join

#for first image loading for shape
import imageio
import numpy as np
import time

class imageDirectoryLoader:
	def __init__(self, batchsize, dirname,subdir=True, name="", coordinator = None):
		session = tf.get_default_session()        
		print "BATCH SIZE = ",batchsize
		with tf.variable_scope("Image_Loader_"+name):
			with tf.device("/cpu:0"):
				if not subdir:
					files=[dirname + f for f in listdir(dirname) if isfile(join(dirname, f))]
					numImages = len(files)
					files = tf.constant(files)

				else:
					dirs = [dirname + f + "/" for f in listdir(dirname) if not isfile(join(dirname, f))]
					images = {}
					files = []
					i=0
					labels = []
					for d in dirs:
						images[d] = [d + f for f in listdir(d) if isfile(join(d, f))]
						files+=images[d]
						for _ in range(len(images[d])):
							labels.append(i)
						i+=1
					numImages=len(labels)
					if len(files)==0:
						raise Exception("Empty directory !")
					files = tf.constant(files)
					labels = tf.constant(labels)
				def genIndexs(size, maxConsec=4):
					# print "I am called"
					ret = []
					while len(ret)<size:
						r = np.random.randint(0, high=numImages)
						c = np.random.randint(1, high=maxConsec)
						for i in range(c):
							if len(ret)==size or r+i >= numImages:
								break
							ret.append(r+i)
					if len(ret) != size:
						# print "I am ERROR"
						exit(-44)
					return np.array(ret, dtype=np.int32)
				# indexs = tf.to_int32(tf.random_uniform([batchsize*4], maxval=int(files.get_shape()[-1])-1))
				indexs=tf.py_func(genIndexs, [batchsize*4], tf.int32)
				
				# indexs = tf.Print(indexs, [indexs])
				randomGather = tf.gather(files, indexs)
				#labelGather = tf.gather(labels, indexs)
				# randomGather = tf.Print(randomGather, [randomGather])

				op =  lambda x : tf.image.decode_image(tf.read_file(x), channels=3)    
				
				read = tf.map_fn(op, randomGather,dtype=tf.uint8)
				# read = tf.Print(read, [tf.shape(read)])
				#read = tf.map_fn(, raw, dtype=tf.uint8)
				read = tf.to_float(read)/256.0
				#read = tf.Print(read, [tf.shape(randomGather)])
				print "*********************************************************************************************"
				# first_img = misc.imread(images[dirs[0]][0])
				first_img = imageio.imread(session.run(files[0]))
				self.shape = first_img.shape
				print(self.shape)
				#fifo = tf.FIFOQueue(batchsize*4, [tf.float32, tf.int32], shapes= [[self.shape[0], self.shape[1], self.shape[2]], []])
				fifo = tf.FIFOQueue(batchsize * 4, [tf.float32],
									shapes=[[self.shape[0], self.shape[1], self.shape[2]]])
				self.batch = fifo.dequeue_many(batchsize)
				# self.batch = tf.Print(self.batch, [fifo.size()])
				#self.batch = tf.Print(self.batch, [self.batch])

				#self.batch = read
				#reshape = lambda x : tf.reshape(x, [self.shape[0], self.shape[1], self.shape[2]])
				#read = tf.reshape(read, [batchsize*2, self.shape[0], self.shape[1], self.shape[2]]) #tf.map_fn(reshape, read, name="reshape")
				#print [self.shape[0], self.shape[1], self.shape[2]], read.get_shape()
				enqueue_op = fifo.enqueue_many([read])#, labelGather])
				qr = tf.train.QueueRunner(fifo, [enqueue_op] * 6)
				self.coord = coordinator if coordinator!= None else tf.train.Coordinator()

				self.threads = qr.create_threads(session, coord=self.coord, daemon=True, start=True)
			
	def stop(self, waitForStop=True):
		self.coord.request_stop()
		if waitForStop:
			self.coord.join(self.threads)
	def getBatch(self):
		return self.batch
	def getLabel(self):
		return 0# self.labels
	def getShape(self):
		return self.shape
            
            

