#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


tf.set_random_seed(1)
np.random.seed(1)

# fake data
r = np.linspace(0.1, 2.7, 100)[:, np.newaxis]          # shape (100, 1)
# noise = np.random.normal(-0.01, 0.01, size=x.shape)
# [rosi,bo1,bo2,bosiw1,bosib1]
# [3.3834546, 1.1936723, 4.200491, -4.7176676, 3.9001603]
# [3.5740135, 1.0703207, 4.3786235, -6.389729, 4.2174034]
# [3.548378, 1.0895519, 4.424833, -6.287387, 4.318791]
# [2.7009578, 1.5794235, 4.147349, -6.329513, 4.3710465]
rosi_,bo1_,bo2_,bosiw1_,bosib1_= [1.3829, -0.5123,  5.9026227, -2.8829744,5.9026227]
boo   =  0.12*np.exp(bo1_*(r/rosi_)**bo2_) 

rosi_,bo1_,bo2_,bosiw1_,bosib1= [3.3834546, 1.1936723, 4.200491, -4.7176676, 3.9001603]

rosi   = tf.Variable(rosi_)
bo1    = tf.Variable(bo1_)
bo2    = tf.Variable(bo2_)
bosiw1 = tf.Variable(bosiw1_)
bosib1 = tf.Variable(bosib1_)
# bosiw2 = tf.Variable(bosiw2_)
# bosib2 = tf.Variable(bosib2_)

# neural network layers
l1 = tf.exp(bo1*(r/rosi)**bo2)
bo = tf.sigmoid(bosiw1*l1+bosib1)

loss = tf.losses.mean_squared_error(boo, bo)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=2.0)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting

for step in range(5000):
    # train and net output
    _, l, bo_ = sess.run([train_op, loss, bo])
    if step % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(r,boo)
        plt.plot(r,bo_, 'r-', lw=3)
        plt.text(0.5, 0, 'Step: %d Loss=%.4f' %(step,l), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.savefig('bo_l2.eps')
plt.show()


# w1_,b1_,w2_,b2_ = sess.run([w1_,b1_,w2_,b2_])

print(sess.run([rosi,bo1,bo2,bosiw1,bosib1]))


