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
tf.compat.v1.disable_eager_execution()


def get_bond_energy(p,bd,bond_data):
    ''' compute bond-energy '''
    unit = 4.3364432032e-2
    bsi  = bond_data[:,0]
    bpi  = bond_data[:,1]
    bpp  = bond_data[:,2]
    
    powb = np.power(np.where(bsi>0.00000001,bsi,0.00000001),p['be2_'+bd])
    expb = np.exp(p['be1_'+bd],(1.0-powb))
    #print(p['Desi_'+bd])
    e_si = p['Desi_'+bd]*bsi*expb*unit
    e_pi = p['Depi_'+bd]*bpi*unit
    e_pp = p['Depp_'+bd]*bpp*unit
    e    = (e_si+e_pi+e_pp)/(p['Desi_'+bd]*unit)
    return e


def fit(step=1000,obj='BO'):
    with open('ffield.json','r') as lf:
         j  = js.load(lf)
         p = j['p']
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
        e      = np.array(E[bd])
        # print(bp)
        # bo     = bp[:,1]
        B_     = np.array(B[bd])
        E_[bd] = get_bond_energy(p,bd,B_)

    train(Bp,D,B_,E_,bonds=bonds,step=step,fitobj=obj)

   
if __name__ == '__main__':
   help_= 'Run with commond: ./init_parameter_fit.py  --o=BE --s=3000 '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--step',default=10000,type=int, help='training steps')
   parser.add_argument('--o',default='BE',type=str, help='fit object BE or BO')
   args = parser.parse_args(sys.argv[1:])
   
   fit(step=args.step,obj=args.o)



