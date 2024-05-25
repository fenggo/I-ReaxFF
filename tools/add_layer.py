#!/usr/bin/env python
import tensorflow as tf
from irff.ml.data import get_data,get_bond_data,get_md_data
from irff.ml.fit import train as train_nn
from train import dataset
tf.compat.v1.disable_eager_execution()


'''
   Add neural network layer for current MLP models
'''


bonds = ['C-C'] 
D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,
                      message_function=3,
                      ffield='ffieldData.json')

train_nn(Bp,D,B,E,bonds=bonds,step=100000,fitobj='BO',
        layer=(9,6),              
        learning_rate=0.0001)






