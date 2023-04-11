#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
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



colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','agua','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']



def plea(direcs={'cwd':getcwd()},
          batch_size=200):
    for m in direcs:
        mol = m
    rn = ReaxFF(libfile='ffield',direcs=direcs,dft='siesta',
                 optword='all',
                 batch_size=batch_size,
                 rc_scale='none',
                 clip_op=False,
                 interactive=True) 
    molecules = rn.initialize()
    rn.session(learning_rate=1.0e-10,method='AdamOptimizer')

    D          = rn.get_value(rn.D)
 
    Eov        = rn.get_value(rn.EOV)
    Eun        = rn.get_value(rn.EUN)
    Elp        = rn.get_value(rn.EL)

    atlab      = rn.lk.atlab
    natom      = molecules[mol].natom

    cell       = rn.cell[mol]
    
    nb_ = 100
    for sp in rn.spec:
        if rn.nsp[sp]>0:
           plt.figure()             # temperature
           plt.ylabel(r"Distribution Density")
           plt.xlabel(r"Atomic energy (unit: eV)")

           Elp_ = np.reshape(Elp[sp],[-1])
           max_ = np.max(Elp_)
           min_ = np.min(Elp_)
           if min_<-0.00001:
              hist,bin_ = np.histogram(Elp_,range=(min_,max_),bins=nb_,density=True)
              plt.plot(bin_[:-1],hist,alpha=0.5,color='red',linestyle='-',
                       label=r"lone pair of %s" %bd)

           Eov_ = np.reshape(Eov[sp],[-1])
           max_ = np.max(Eov_)
           min_ = np.min(Eov_)
           if min_<-0.00001:
              hist,bin_ = np.histogram(Eov_,range=(min_,max_),bins=nb_,density=True)
              plt.plot(bin_[:-1],hist,alpha=0.5,color='green',linestyle='--',
                       label=r"over energy of %s" %sp)

           Eun_ = np.reshape(Eun[sp],[-1])
           max_ = np.max(Eun_)
           min_ = np.min(Eun_)
           if min_<-0.00001:
              hist,bin_ = np.histogram(Eun_,range=(min_,max_),bins=nb_,density=True)
              plt.plot(bin_[:-1],hist,alpha=0.5,color='blue',linestyle='-.',
                       label=r"under energy of %s" %sp)

           plt.legend(loc='best')
           plt.savefig('eatomic_%s.eps' %sp) 
           plt.close() 

           print('\n -- %s -- \n' %sp)
           for i in range(rn.nsp[sp]):
               # if Elp[sp][i][0]>0.0:
               print('-  %s D:' %sp,D[sp][i][0],
                     '-  %s Elp:' %sp,Elp[sp][i][0],
                     '-  %s Eov:' %sp,Eov[sp][i][0],
                     '-  %s Eun:' %sp,Eun[sp][i][0])



if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   
   plea(direcs={'ethane':'/home/feng/siesta/train2/ethanee'})
