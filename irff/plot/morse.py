#!/usr/bin/env python
from os.path import isfile
import tensorflow as tf
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import json as js


def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s

def morseX(r,a=0.90,re=1.098):
    y = np.exp(-(r-re)/a)
    return y


def morseY(r,a=1.0,b=1.5,re=1.098):
    y = np.exp(-(r-re)/a-((r-re)**2.0)/b)
    return y


def plotMorse():
    x    = np.linspace(0.0,5.0,50)
    x_   = np.linspace(0.0,5.0,50,dtype=np.float32)[:, np.newaxis]  
    s1   = morseX(x)
    s2   = morseY(x)
    s3   = morseX(x,a=0.37)
    NNa  = Linear(J='WBa.json')
    sNNa = NNa(x_)  
    # sNNa = np.squeeze(sNNa)
    NNb  = Linear(J='WBb.json')
    sNNb = NNb(x_)  
    # sNNb = np.squeeze(sNNb)
    a  = 0.90
 
    plt.figure() 
    plt.ylabel('X, Y and NN')
    plt.xlabel(r'r($\AA$)')
    plt.xlim(0,5.0)
    plt.ylim(0,3.0)

    plt.plot(x,s1,label=r'$X,a=0.90$ $\AA$',
             color='black', linewidth=1, linestyle='-')
    plt.plot(x,s3,label=r'$X,a=0.37$ $\AA$',
             color='blue', linewidth=1, linestyle='-')
    plt.plot(x,s2,label=r'$Y,a=1.0$ $\AA$ and $b=1.5$ $\AA^2$',
             color='red', linewidth=1, linestyle='-')

    plt.scatter(x,sNNa,label=r'$NeuralNetwork-fit$ $to$ $X$',
                marker = 'o',color='r',alpha=0.75,s=10)
    plt.scatter(x,sNNb,label=r'$NeuralNetwork-fit$ $to$ $Y$',
                marker = '^',color='b',alpha=0.75,s=10)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('morse.pdf') 
    plt.close()


class Linear(tf.keras.Model):
    def __init__(self,J=None):
        super().__init__()
        if J is None:
           self.wi  = tf.Variable(tf.random.normal([1,8],stddev=0.2),name='wi')
           self.bi  = tf.Variable(tf.random.normal([8],stddev=0.2),name='bi')
           self.wh  = tf.Variable(tf.random.normal([8,8],stddev=0.2),name='wh')
           self.bh  = tf.Variable(tf.random.normal([8],stddev=0.2),name='bh')
           # self.wh1 = tf.Variable(tf.random.normal([8,8],stddev=0.2),name='wh1')
           # self.bh1 = tf.Variable(tf.random.normal([8],stddev=0.2),name='bh1')
           self.wo  = tf.Variable(tf.random.normal([8,1],stddev=0.2),name='wo')
           self.bo  = tf.Variable(tf.random.normal([1],stddev=0.2),name='bo')
        else:
           with open(J,'r') as lf:
                j = js.load(lf)
           self.wi  = tf.Variable(j['wi'],name='wi')
           self.bi  = tf.Variable(j['bi'],name='bi')
           self.wh  = tf.Variable(j['wh'],name='wh')
           self.bh  = tf.Variable(j['bh'],name='bh')
           # self.wh1 = tf.Variable(j['wh1'],name='wh1')
           # self.bh1 = tf.Variable(j['bh1'],name='bh1')
           self.wo  = tf.Variable(j['wo'],name='wo')
           self.bo  = tf.Variable(j['bo'],name='bo')


    def call(self, r):
        i_layer  = tf.sigmoid(tf.matmul(r,self.wi)+self.bi)
        h_layer  = tf.sigmoid(tf.matmul(i_layer,self.wh)+self.bh)
        # h_layer  = tf.sigmoid(tf.matmul(h0_layer,self.wh1)+self.bh1)
        output   = tf.sigmoid(tf.matmul(h_layer,self.wo)+self.bo)*10.0
        # loss = tf.losses.mean_squared_error(o_layer, y)   # compute cost
        return output


    def save(self):
        with open('WeightAndBias.json','w') as fj:
             j = {}
             j['wi'] = self.wi.numpy().tolist()
             j['bi'] = self.bi.numpy().tolist()
             j['wh'] = self.wh.numpy().tolist()
             j['bh'] = self.bh.numpy().tolist()
             j['wo'] = self.wo.numpy().tolist()
             j['bo'] = self.bo.numpy().tolist()
             js.dump(j,fj,sort_keys=True,indent=2)


def fitMorse(func):
    x   = np.linspace(0.0,5.0,100,dtype=np.float32)[:, np.newaxis]  
    y   = func(x)

    # neural network layers
    # model = Linear(J='WeightAndBias.json')
    model = Linear(J='WBa.json')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    plt.ion()   # something about plotting


    for step in range(5000):
        # train and net output
        with tf.GradientTape() as tape:
            y_pred = model(x)     
            loss = tf.reduce_mean(tf.square(y_pred - y))
        
        grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        
        if step % 10 == 0:
           plt.cla()
           plt.scatter(x,y)
           plt.plot(x,y_pred, 'r-', lw=3)
           plt.text(0.5, 0, 'Step: %d Loss=%.4f' %(step,loss), fontdict={'size': 20, 'color': 'red'})
           # plt.pause(0.1)

        if loss<0.0001:
           break

    plt.savefig('FitMorse.pdf')
    plt.show()
    model.save()


def func(x):
    s = sigmoid(x)
    return (s*s - 0.5)**2


def resolve():
    res = minimize_scalar(func,method='brent')
    print(res.x)
  
    t = sigmoid(0.8813735870089523)
    print(t*t)


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   plotMorse()
   # fitMorse(morseX)


   