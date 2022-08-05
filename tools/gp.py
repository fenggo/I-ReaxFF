#!/usr/bin/env python
from distutils.command.build import build
from os.path import isfile
from os import system
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
from irff.ml.gpdata import get_data,get_bond_data,get_md_data
from irff.ml.gpfit import train
tf.compat.v1.disable_eager_execution()


def bo(i=4,j=1,traj='md.traj',bonds=None):
    D, Bp, B, R, E = get_bond_data(i,j,images=None, traj=traj,bonds=bonds)
    for i,bp in enumerate(Bp):
        print('step: {:4d} R: {:6.4f} Di: {:6.4f} Bij: {:6.4f} Dj: {:6.4f} '
              'B: {:6.4f} E: {:6.4f}'.format(i,
              R[i],D[i][0],D[i][1],D[i][2],np.sum(B[i]),E[i]))
    print('\n r & E:',R[i],E[i])
    print('\n B\': \n',Bp[i])
    print('\n D: \n',D[i])
    print('\n B: \n',B[i])


def fit(step=1000,obj='BO'):
    dataset = {'h22-v':'aimd_h22/h22-v.traj',
                'dia-0':'data/dia-0.traj',
                'gp2-0':'data/gp2-0.traj',
                'gp2-1':'data/gp2-1.traj',
                }

    trajdata = ColData()
    strucs = ['h22',
            'ch2',
            'cn2',
            #'c6',
            #'c10',
            #'ch4',
            'c2h4',
            'c2h6',
            'c2h8',
            'c3h8',
            'ch3nh2',
            'n2h4',
            'nh2',
            'c2h6nh2',
            'no2',
            'o2n',
            'ch3no2',
            'o22',
            'n22',
            'oh2',
            'h2o',
            'nh3',
            'co2',
            'hmx1',
            'hmx2',
            ]
    batchs = {'others':50}

    for mol in strucs:
        b = batchs[mol] if mol in batchs else batchs['others']
        trajs = trajdata(label=mol,batch=b)
        dataset.update(trajs)

    bonds = ['H-N']
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')


    # for bd in bonds:        ## 高斯模型  
    #     gp.fit(D[bd],E[bd])
    

    #D_,Bp_,B_,R_,E_ = get_md_data(traj='md.traj',bonds=bonds,ffield='ffield.json')
    

    # for bd in B:
    #     B[bd].extend(B_)
    #     Bp[bd].extend(Bp_)
    #     D[bd].extend(D_)
    #     E_
    #     E_[bd] = gp.predict()

    ## 高斯模型  
    #  D  E 
    for bd in bonds:
        # if bd == 'H-N':
        E[bd] = np.array(E[bd]) * 1.5

    train(Bp,D,B,E,step=step,fitobj=obj,bonds=bonds)

   
if __name__ == '__main__':
   ''' Run with commond: 
      ./gp.py fit --o=BE --s=3000 
      ./gp.py bo  --t=data/n2h4-0.traj --i=2 --j=1 
      '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [fit,bo])
   argh.dispatch(parser)  



