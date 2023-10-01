#!/usr/bin/env python
from distutils.command.build import build
from os.path import isfile
from os import system
import sys
import argh
import argparse
import json as js
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from irff.data.ColData import ColData
from irff.ml.data import get_data,get_bond_data,get_md_data
from irff.ml.fit import train
tf.compat.v1.disable_eager_execution()


def bo(i=4,j=1,traj='md.traj',bonds=None):
    D, Bp, B, R, E = get_bond_data(i,j,images=None, traj=traj,bonds=bonds)
    for i,bp in enumerate(Bp):
        print(i,R[i],D[i][1],np.sum(B[i]),E[i])
    print('\n r & E:',R[i],E[i])
    print('\n B\': \n',Bp[i])
    print('\n D: \n',D[i])
    print('\n B: \n',B[i])


def fit(step=1000,obj='BO'):
    unit =  4.3364432032e-2
    Desi = 424.95
    dataset = {'al60-0': 'data/al64-0.traj',
               'al60-1': 'data/al64-1.traj',
               'AlO-0': 'data/AlO-0.traj',
               'AlO-1': 'data/AlO-1.traj', 
               }
    trajdata = ColData()

    strucs = [ ]
    batchs = {'others':50}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['Al-Al','O-Al']
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffield.json')
    E_ = {}
    B_ = {}
    for bd in B:
        bp     = np.array(Bp[bd])
        # print(bp)
        bo     = bp[:,1]
        m_     = max(bo)
        E_[bd] = bo*0.875/m_
        B_[bd] = bp*np.array([0.45,0.425,0.4])

    train(Bp,D,B_,E_,bonds=bonds,step=step,fitobj=obj)

   
if __name__ == '__main__':
   help_= 'Run with commond: ./init_parameter_fit.py  --o=BE --s=3000 '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--step',default=10000,type=int, help='training steps')
   parser.add_argument('--o',default='BE',type=str, help='fit object BE or BO')
   args = parser.parse_args(sys.argv[1:])
   
   fit(step=args.step,obj=args.o)



