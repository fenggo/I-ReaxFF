#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def morse(r,rosi=0.6,alfa=0.5):
    r_        = r/rosi
    mors_exp1 = np.exp(alfa*(r_-1.0))  ## Morse should be: 1.0 - r
    mors_exp2 = np.square(mors_exp1) 
    emorse    = mors_exp2 - 2.0*mors_exp1
    return emorse


def plot_morse():
    x    = np.linspace(0.0,1.2,100)
    s    = morse(x)
    # for r,m in zip(x,s):
    #     print(r,m)
    # print(np.min(s),np.max(s))

    plt.figure()
    plt.ylabel(r'$Morse Potential$')
    plt.xlabel(r'$x$')

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data', 0))

    plt.plot(x,s,label=r'$Morse$',
             color='b', linewidth=2, linestyle='-')

    plt.legend()
    plt.savefig('Morse.pdf') 
    plt.close()


def func(x):
    s = morse(x)
    return s**2


def resolve():
    res = minimize_scalar(func,method='brent')
    print(res.x)
  
    #t = morse(0.8813735870089523)
    #print(t*t)


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   plot_morse()
   # resolve()

   