import tensorflow as tf
import numpy as np

from NetBluePrint.core.dataset import dataset
from googlefinance.client import get_price_data, get_prices_data, get_prices_time_data

class trading_simple(dataset):
    def __init__(self,batchsize=64, stock=".DJI", data_granularity=86400, dataspan="3Y", batch_length=7, prediction_length=1):
        super(trading_simple, self).__init__(batchsize)

        param = {
            'q': stock,  # Stock symbol (ex: "AAPL")
            'i': data_granularity,  # Interval size in seconds ("86400" = 1 day intervals)
            'x': "INDEXDJX",  # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': dataspan  # Period (Ex: "1Y" = 1 year)
        }
        print(get_price_data(param))
        self.data = np.array(get_price_data(param).values[:,:-1], dtype=np.float32)

        with tf.name_scope("trading_simple"):
            def sample_data():
                rand = np.random.randint(0, high=self.data.shape[0]-(batch_length+prediction_length), size=self.batchsize)
                expanded_rand= np.empty(((batch_length+prediction_length)*self.batchsize), dtype=np.int32)
                for i in range(batch_length+prediction_length):
                    expanded_rand[i::(batch_length+prediction_length)]=rand+i
                values = np.take(self.data, expanded_rand,axis=0)
                values = np.reshape(values, [self.batchsize, (batch_length+prediction_length), self.data.shape[1]])

                mean_values=np.mean(values, axis=0)
                values=values-mean_values

                feed_sequence=values[:,:batch_length,:]
                outcome_sequence = values[:,batch_length:,:]
                return feed_sequence,outcome_sequence

            feed, outcome = tf.py_func(sample_data, [], [np.float32, np.float32])
            feed = tf.reshape(feed, [self.batchsize, batch_length, self.data.shape[-1]])
            outcome = tf.reshape(outcome, [self.batchsize, prediction_length, self.data.shape[-1]])

            self.data_dict["feed"]=feed
            self.data_dict["outcome"]=outcome