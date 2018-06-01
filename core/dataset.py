# -*- coding: utf-8 -*-
"""
Created on Thuesday May 31 21:38:32 2018

@author: arnaud
"""


class dataset(object):
    def __init__(self, batchsize):
        self.data_dict = {}
        self.batchsize = batchsize

    def __getitem__(self, key):
        return self.data_dict[key]
    def get_keys(self):
        return self.data_dict.keys()
