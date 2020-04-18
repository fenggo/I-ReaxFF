#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


tf.set_random_seed(1)
np.random.seed(1)

# fake data
r = np.linspace(0.1, 2.7, 100,dtype=np.float32)[:, np.newaxis]          # shape (100, 1)
print('-  data shape:',r.shape)

rosi_,bo1_,bo2_,bosiw1_,bosib1_= [1.3829, -0.5123,  5.9026227, -2.8829744,5.9026227]
y   =  0.12*np.exp(bo1_*(r/rosi_)**bo2_) 

wi  = tf.Variable(tf.random.normal([1,8],stddev=0.2),name='w_input_layer')
bi  = tf.Variable(tf.random.normal([8],stddev=0.2),name='b_input_layer')
wh  = tf.Variable(tf.random.normal([8,8],stddev=0.2),name='w_hiden_layer')
bh  = tf.Variable(tf.random.normal([8],stddev=0.2),name='b_hiden_layer')
wo  = tf.Variable(tf.random.normal([8,1],stddev=0.2),name='w_output_layer')
bo  = tf.Variable(tf.random.normal([1],stddev=0.2),name='b_output_layer')

# neural network layers

i_layer = tf.sigmoid(tf.matmul(r,wi)+bi)
h_layer = tf.sigmoid(tf.matmul(i_layer,wh)+bh)
o_layer = tf.sigmoid(tf.matmul(h_layer,wo)+bo)


loss = tf.losses.mean_squared_error(o_layer, y)   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=2.0)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

plt.ion()   # something about plotting

for step in range(50000):
    # train and net output
    _, l, y_ = sess.run([train_op, loss, o_layer])
    if step % 10 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(r,y)
        plt.plot(r,y_, 'r-', lw=3)
        plt.text(0.5, 0, 'Step: %d Loss=%.4f' %(step,l), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.savefig('bo_nn.eps')
plt.show()



