#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.reaxfflib import read_lib,write_lib
from irff.reax import ReaxFF
import argh
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab  
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from ase.io.trajectory import Trajectory
import tensorflow as tf

colors = ['deepskyblue','fuchsia','chartreuse','darkviolet',
          'midnightblue','red','deeppink','agua','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','darkcyan',
          'firebrick','mediumslateblue','khaki','gold','k']


def plov(direcs={'ethane':'/home/gfeng/siesta/train/ethane'},
         atoms=[8,51],
         batch_size=2000):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 clip_op=False,
                 pkl=False,
                 InitCheck=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    atom_name = molecules[mol].atom_name

    D     = rn.get_value(rn.D)
    Dp    = rn.get_value(rn.Delta_lpcorr)
    Dl    = rn.get_value(rn.Delta_lp)
    De    = rn.get_value(rn.Delta_e)
    NLP   = rn.get_value(rn.nlp)
    Eov   = rn.get_value(rn.EOV)

    atlab = rn.lk.atlab
    natom = molecules[mol].natom
    d     = np.zeros([natom,batch_size])
    dlc   = np.zeros([natom,batch_size])
    dl    = np.zeros([natom,batch_size])
    de    = np.zeros([natom,batch_size])
    nlp   = np.zeros([natom,batch_size])
    eov   = np.zeros([natom,batch_size])
    cell  = rn.cell[mol]
    p     = rn.p_

    for sp in rn.spec:
        if rn.nsp[sp]>0:
           for l,lab in enumerate(atlab[sp]):
               if lab[0]==mol:
                  i = int(lab[1])
                  d[i]   = D[sp][l]
                  dlc[i] = Dp[sp][l]
                  dl[i]  = Dl[sp][l]
                  de[i]  = 2.0*De[sp][l]
                  nlp[i] = NLP[sp][l]
                  eov[i] = Eov[sp][l]

    plt.figure()      

    plt.subplot(3,2,1)    
    # plt.ylabel(r'$\Delta_{lpcorr}$')
    # plt.xlabel(r"Step")
    
    for i,atm in enumerate(atoms):
        plt.plot(d[atm],alpha=0.5,color=colors[(i)%len(colors)],
                 label=r"$\Delta$@%s:%d" %(atom_name[atm],atm))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,2)    
    # plt.ylabel(r'$\Delta_{lpcorr}$')
    # plt.xlabel(r"Step")
    
    for i,atm in enumerate(atoms):
        plt.plot(dlc[atm],alpha=0.5,color=colors[(i)%len(colors)],
                 label=r"$\Delta_{lpcorr}$@%s:%d" %(atom_name[atm],atm))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,3)    
    # plt.ylabel(r'$\Delta_{lp}$')
    # plt.xlabel(r"Step")
    for i,atm in enumerate(atoms):
        plt.plot(dl[atm],alpha=0.5,color=colors[(i)%len(colors)],
                 label=r"$\Delta_{lp}$@%s:%d" %(atom_name[atm],atm))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,4)    
    # plt.ylabel(r'$\Delta_e$')
    plt.xlabel(r"Step")
    for i,atm in enumerate(atoms):
        plt.plot(de[atm],alpha=0.5,color=colors[(i)%len(colors)],
                 label=r'$\Delta_e$@%s:%d' %(atom_name[atm],atm))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,5)    
    # plt.ylabel(r'$Eover$')
    plt.xlabel(r"Step")
    for i,atm in enumerate(atoms):
        plt.plot(nlp[atm],alpha=0.5,color=colors[(i)%len(colors)],
                 label=r"$NLP$@%s:%d" %(atom_name[atm],atm))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.subplot(3,2,6)    
    # plt.ylabel(r'$Eover$')
    plt.xlabel(r"Step")
    for i,atm in enumerate(atoms):
        plt.plot(dlc[atm]+p['val_'+atom_name[atm]],alpha=0.5,color=colors[(i)%len(colors)],
                 label=r"$\Delta_{lpcorr}+val$@%s:%d" %(atom_name[atm],atm))
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Delta_lpcorr.eps',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./pv.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   plov()

