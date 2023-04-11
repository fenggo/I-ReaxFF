#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def sigmoid(x):
    # sigmoid_alpha = 1.0
    # s = np.log(1.0+np.exp(sigmoid_alpha*x))/sigmoid_alpha
    s = 1.0/(1.0+np.exp(-x))
    return s


def plot_s():
    x    = np.linspace(-5.0,5.0,100)
    s    = sigmoid(x)

    plt.figure()
    plt.ylabel(r'$Sigmoid$')
    plt.xlabel(r'$x$')

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data', 0))

    plt.plot(x,s,label=r'$Sigmoid$ $Neuron$',
             color='b', linewidth=2, linestyle='-')

    plt.legend()
    plt.savefig('sigmoid.eps') 
    plt.close()


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
   plot_s()


   