# -*- coding: utf-8 -*-
"""
Created on Thuesday May 31 21:38:32 2018

@author: arnaud
"""


class dataset(object):
    def __init__(self, construct_log, batchsize):
        self.data_dict = {}
        self.construct_log = construct_log
        self.batchsize = batchsize
        construct_log["batchsize"]=batchsize

    def __getitem__(self, key):
        return self.data_dict[key]
    def get_keys(self):
        return self.data_dict.keys()
    def get_datadict(self):
        return self.data_dict
