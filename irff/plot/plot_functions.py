#!/usr/bin/env python
import argh
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def morse(r,rosi=0.6,alfa=0.3):
    r_         = r/rosi
    mors_exp1  = np.exp(alfa*(1.0-r_))  ## Morse should be: 1.0 - r
    mors_exp2  = np.square(mors_exp1) 


    mors_exp1l= np.exp(alfa*-0.3)
    mors_exp2l= np.square(mors_exp1l) 
    emorse_l   = mors_exp2l - 2.0*mors_exp1l 

    emorse     = 2.0*mors_exp1 - mors_exp2  + mors_exp2l - 2.0*mors_exp1l 
    # emorse     = mors_exp2 - 2.0*mors_exp1 - mors_exp20 + 2.0*mors_exp10
    return emorse

def morse0(r,rosi=0.6,alfa=1.5):
    r_         = r/rosi
    mors_exp1  = np.exp(alfa*(1.0-r_))  ## Morse should be: 1.0 - r
    mors_exp2  = np.square(mors_exp1) 
    emorse     = mors_exp2 - 2.0*mors_exp1 
    return emorse

def dmorse(r,rosi=0.6,alfa=0.12):
    r_        = r/rosi
    mors_exp1 = -alfa*np.exp(alfa*(1.0-r_))/rosi  ## Morse should be: 1.0 - r
    mors_exp2 = 2.0*np.square(mors_exp1) *alfa*np.exp(alfa*(r_-1.0))/rosi
    emorse    = mors_exp2 - 2.0*mors_exp1
    return emorse

def fbo(r,bo1=-0.04,bo2=3.5,rosi=0.65):
    b = np.exp(bo1*(r/rosi)**bo2)
    return b


def mors():
    x    = np.linspace(0.0,1.2,100)
    s    = morse(x)
    # for r,m in zip(x,s):
    #     print(r,m)
    print(np.min(s),np.max(s))

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

def bo():
    x    = np.linspace(0.0,2.5,100)
    s    = fbo(x)
    # for r,m in zip(x,s):
    #     print(r,m)
    # print(np.min(s),np.max(s))
    plt.figure()
    plt.ylabel(r'$Border-Order Uncorrected$')
    plt.xlabel(r'$x$')

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data', 0))

    plt.plot(x,s,label=r'$BOP$',
             color='b', linewidth=2, linestyle='-')

    plt.legend()
    plt.savefig('bop.pdf') 
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
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [mors,bo])
   argh.dispatch(parser)

   