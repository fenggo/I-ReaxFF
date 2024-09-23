#!/usr/bin/env python
import sys
#import argh
import argparse
import json as js
import tensorflow as tf
import numpy as np
#import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
#from scipy.optimize import minimize_scalar
from irff.data.ColData import ColData
from irff.ml.data import get_data,get_bond_data # ,get_md_data
from irff.ml.fit import train
from irff.ml.reax_funcs import get_bond_energy
tf.compat.v1.disable_eager_execution()



def fit(step=1000,obj='BO'):
    with open('ffieldData.json','r') as lf:
         j  = js.load(lf)
         p = j['p']

    dataset = { }

    trajdata = ColData()
    strucs = ['n29',
          #'si64',
          #'si3n4',
          'si4h12',
          'si2h6',
          ]
    batchs = {'others':500}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['H-H','H-Si','Si-Si','N-N'] 
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    E_ = {}
    # B_ = {}
    for bd in B:
        bp     = np.array(Bp[bd])
        e      = np.array(E[bd])
        # print(bp)
        # bo     = bp[:,1]
        B_     = np.array(B[bd])
        E_[bd] = get_bond_energy(p,bd,B_)
      

    train(Bp,D,B,E_,bonds=bonds,step=step,fitobj=obj)

   
if __name__ == '__main__':
   help_= 'Run with commond: ./init_parameter_fit.py  --o=BE --s=3000 '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--step',default=10000,type=int, help='training steps')
   parser.add_argument('--o',default='BE',type=str, help='fit object BE or BO')
   args = parser.parse_args(sys.argv[1:])
   
   fit(step=args.step,obj=args.o)

