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
from irff.ml.reax_funcs import reax_bond_energy
from irff.reax_force_data import reax_force_data
tf.compat.v1.disable_eager_execution()

def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           # print(kk)
           if len(kk)==2:
              if kk[0]!=kk[1]:
                 offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs
def fit(step=1000,obj='BO',random_init=0,learning_rate=0.01):
    with open('ffieldData.json','r') as lf:
         j  = js.load(lf)
         p = j['p']
         m = j['m']
    spec,bonds,offd,angs,torp,hbs = init_bonds(p)
    dataset = { }

    trajdata = ColData()
    strucs = ['n29',
              'si64',
              'si3n4',
              'si4h12',
              'si2h6',
              ]
    batchs = {'others':500}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['H-H','H-Si','Si-Si','N-N','N-Si'] 
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')
    
    # for st in dataset:
    #     data_ = reax_force_data(structure=st,
    #                             traj=dataset[st],
    #                             vdwcut=10.0,
    #                             rcut=j['rcut'],
    #                             rcuta=j['rcutBond'],
    #                             hbshort=6.75,
    #                             hblong=7.5,
    #                             batch=1000,
    #                             variable_batch=True,
    #                             m=m,
    #                             mf_layer=j['mf_layer'],
    #                             p=p,spec=spec,bonds=bonds,
    #                             angs=angs,tors=torp,
    #                             hbs=hbs,
    #                             screen=False)

    E_ = {}
    # B_ = {}
    for bd in B:
        bp     = np.array(Bp[bd])
        e      = np.array(E[bd])
        B[bd].append([0.0,0.0,0.0])
        B_     = np.array(B[bd])
        E_[bd] = reax_bond_energy(p,bd,B_)
        # for i,e_ in enumerate(e):
        #     print(e_,E_[bd][i],B_[i])
        # print(np.max(e),p['Desi_'+bd]*unit)
      

    train(Bp,D,B,E_,bonds=bonds,step=step,fitobj=obj,learning_rate=learning_rate,
          layer=(9,1),random_init=random_init)

   
if __name__ == '__main__':
   help_= 'Run with commond: ./init_parameter_fit.py  --o=BE --s=3000 '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--step',default=10000,type=int, help='training steps')
   parser.add_argument('--r',default=0,type=int, help='whether random init')
   parser.add_argument('--l',default=0.01,type=float, help='learning rate')
   parser.add_argument('--o',default='BE',type=str, help='fit object BE or BO')
   args = parser.parse_args(sys.argv[1:])
   
   fit(step=args.step,obj=args.o,random_init=args.r,learning_rate=args.l)



