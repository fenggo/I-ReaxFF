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
def fit(step=1000,obj='BO',random_init=0,learning_rate=0.01,test=0):
    unit = 4.3364432032e-2
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
    
    if test:
       D,Bp,B,R,E = get_data(dataset={'md':'md.traj'},bonds=bonds,ffield='ffieldData.json')
       D2,Bp2,B2,R2,E2 = get_data(dataset={'md':'md.traj'},bonds=bonds,ffield='ffield.json')
       for bd in E:
           E1_ = E[bd]
           E2_ = E2[bd]
           if not E[bd]:
              continue
           bmax= np.max(B[bd])
           print('\n ------ max bo ------ \n',bmax)
           for i,e in enumerate(E1_):
               print('B({:.8f} {:.8f} {:.8f})'.format(B[bd][i][0],B[bd][i][1],B[bd][i][2]),
                     'E: ',E1_[i],E2_[i]*p['Desi_{:s}'.format(bd)]*unit)
    else:
       D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    E_ = {}
    b1 = np.random.uniform(0.0,0.4,(10000,3))
    b2 = np.random.uniform(0.0,0.1,(10000,3))
    b3 = np.random.normal(loc=[0.1,0.05,0.03],scale=[0.05,0.05,0.05],size=[10000,3])
    b3 = np.where(b3>0.0,b3,0.0)

    for bd in B:
        bp     = np.array(Bp[bd])
        e      = np.array(E[bd])
        B[bd].append([0.0,0.0,0.0])
        B[bd].extend(b1.tolist())
        B[bd].extend(b2.tolist())
        B[bd].extend(b3.tolist())
        B_     = np.array(B[bd])
        bmax   = np.max(B_)
        print('\n ------ max bo ------ \n',bmax)
        E_[bd] = reax_bond_energy(p,bd,B_)
        # for i,e_ in enumerate(e):
        #     print(e_,E_[bd][i],B_[i])
        # print(np.max(e),p['Desi_'+bd]*unit)
      
    if not test:
       train(Bp,D,B,E_,bonds=bonds,step=step,fitobj=obj,learning_rate=learning_rate,
             layer=(9,1),random_init=random_init)

   
if __name__ == '__main__':
   help_= 'Run with commond: ./init_parameter_fit.py  --o=BE --s=3000 '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--step',default=10000,type=int, help='training steps')
   parser.add_argument('--r',default=0,type=int, help='whether random init')
   parser.add_argument('--t',default=0,type=int, help='whether test result')
   parser.add_argument('--l',default=0.01,type=float, help='learning rate')
   parser.add_argument('--o',default='BE',type=str, help='fit object BE or BO')
   args = parser.parse_args(sys.argv[1:])
   
   fit(step=args.step,obj=args.o,random_init=args.r,learning_rate=args.l,test=args.t)



