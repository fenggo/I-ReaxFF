#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
# import matplotlib
# matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.reaxfflib import read_lib,write_lib
from irff.irff_np import IRFF_NP
from irff.AtomOP import AtomOP
import argh
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.io import read,write
import tensorflow as tf
import json as js


colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']


def sigmoid(x):
    s = 1.0/(1.0+np.exp(-x))
    return s


def f_nn(x,wi,bi,w,b,wo,bo,layer=5):
    X   = np.expand_dims(np.stack(x,axis=0),0)
    # print(X.shape)
    # print(wi.shape)

    o   = []
    o.append(sigmoid(np.matmul(X,wi)+bi))  
                                                                  # input layer
    for l in range(layer):                                        # hidden layer      
        o.append(sigmoid(np.matmul(o[-1],w[l])+b[l]))
    
    o_  = sigmoid(np.matmul(o[-1],wo) + bo) 
    out = np.squeeze(o_)                                          # output layer
    return out


def pltf(atomi,atomj,gen):
    atoms = read(gen)
    ao = AtomOP(atoms)
    # pairs = [[1,2],[13,7],[5,26]]
    pairs = [[0,1]]
    images = ao.stretch(pairs,nbin=50,wtraj=True)
    ao.close()
    # view(images)
    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
    e_,r_ = [],[]
    b_,f_ = [],[]
    for atoms in images:
        ir.calculate(atoms)
        # positions = atoms.get_positions()
        # r = np.sqrt(np.sum(np.square(positions[1]-positions[0])))

        r_.append(ir.r[atomi][atomj])
        e_.append(atoms.get_potential_energy())
        b_.append(ir.H[0][atomi][atomj])
        f_.append(ir.F[-1][atomi][atomj])

    fig, ax = plt.subplots() 
    plt.plot(r_,f_,label=r'$f_{NN}$ vs $BO^t=0$', color='blue', 
             linewidth=2, linestyle='-')
    # plt.plot(b_,f_,label=r'$f_{NN}$ vs $BO^{t=0}$', color='blue', 
    #          linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Estretch.svg') 
    # plt.show()
    plt.close()


def plf3d(atomi,atomj,traj,bo_=2.11):
    images = Trajectory(traj)
    atoms  = images[0]
    # positions  = atoms.get_positions()
    sym        = atoms.get_chemical_symbols()

    with open('ffield.json','r') as lf:
         j = js.load(lf)
         m                   = j['m']

    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
     
    Di_,Dj_,F_,BO_ = [],[],[],[]
    for _,atoms in enumerate(images):
        ir.calculate(atoms)
        Di_.append(ir.D[0][atomi]-ir.H[0][atomi][atomj])
        Dj_.append(ir.D[0][atomj]-ir.H[0][atomi][atomj])
        BO_.append(ir.H[0][atomi][atomj])
        F_.append(ir.F[-1][atomi][atomj])

    for _,d in enumerate(Di_):
        print(Di_[_],Dj_[_],F_[_])

    Di_ =Dj_ = np.arange(1.54,1.62,0.0001)
    Di,Dj    = np.meshgrid(Di_, Dj_)

    i_,j_ = Di.shape
    F   = np.zeros([i_,i_]) 

    # di_= np.expand_dims(ir.D[0],axis=0)-ir.H[0]
    # dj_= np.expand_dims(ir.D[0],axis=1)-ir.H[0]
    # bo_= ir.H[0]
    
    wi = np.array(m['f1wi_C-C'])
    bi = np.array(m['f1bi_C-C'])
    w  = np.array(m['f1w_C-C'])
    b  = np.array(m['f1b_C-C'])
    wo = np.array(m['f1wo_C-C'])
    bo = np.array(m['f1bo_C-C'])

    for i in range(i_):
        for j in range(i,i_):
            di_= Di[i][j]
            dj_= Dj[i][j]

            Fi    = f_nn([dj_,di_,bo_],wi,bi,w,b,wo,bo,layer=ir.bo_layer[1])
            F_     = 2.0*Fi*Fi
            # print(F_)
            F[i][j]=F_
            F[j][i]=F_


    fig = plt.figure()
    ax  = Axes3D(fig)
    # plt.xlabel("Delta'")
    ax  = plt.subplot(111, projection='3d')
    # ax.plot_surface(Di,Dj,F, cmap=plt.get_cmap('rainbow'))
    ax.contourf(Di,Dj,F, zdir='z', cmap=plt.get_cmap('rainbow'))

    plt.savefig('F.svg') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   pltf(0,1,'c2h4.traj')

