#!/usr/bin/env python
import argh
import argparse
import tensorflow as tf
import numpy as np
from irff.data.ColData import ColData
from irff.ml.data import get_data,get_bond_data 
from train import dataset
from irff.ml.fit import train

tf.compat.v1.disable_eager_execution()


def fit(step=1000,obj='BO',i=0,j=1,traj='md.traj',bd=None,dataset={}):
    unit =  4.3364432032e-2
    Desi = 424.95

    bonds = ['C-C','C-H','C-N','H-O','C-O','H-H','H-N','N-N','O-N','O-O'] 
    D,Bp,B,R,E = get_data(dataset=dataset,bonds=bonds,ffield='ffieldData.json')

    D_, Bp_, B_, R_, E_ = get_bond_data(i,j,traj=traj)
    
    # bd = 'N-N'
    for d,bp,b in zip(D_,Bp_,B_):
        D[bd].append(d)
        Bp[bd].append(bp)
        B[bd].append([0.0,0.0,0.0])

    train(Bp,D,B,E,bonds=bonds,step=step,fitobj=obj)

if __name__ == '__main__':
   # parser = argparse.ArgumentParser()
   # argh.add_commands(parser, [fit])
   # argh.dispatch(parser)  
   fit(i=15,j=22,step=3000,obj='BO',dataset=dataset,traj='md.traj',bd='N-N')



