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



def pleb(direcs={'cwd':getcwd()},
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

    rbd        = rn.get_value(rn.rbd)
 
    Esi        = rn.get_value(rn.sieng)
    Epi        = rn.get_value(rn.pieng)
    Epp        = rn.get_value(rn.ppeng)

    atlab      = rn.lk.atlab
    natom      = molecules[mol].natom

    cell       = rn.cell[mol]
    
    nb_ = 100

    for bd in rn.bonds:
        if rn.nbd[bd]>0:
           plt.figure()             # temperature
           plt.ylabel(r"Distribution Density")
           plt.xlabel(r"Bond energy (unit: eV)")

           esi = np.reshape(Esi[bd],[-1])
           max_ = np.max(esi)
           if max_>0.1:
              hist,bin_ = np.histogram(esi,range=(0.1,max_),bins=nb_,density=True)
              plt.plot(bin_[:-1],hist,alpha=0.5,color='red',linestyle='-',
                       label=r"$\sigma$ bond of %s" %bd)

           epi = np.reshape(Epi[bd],[-1])
           max_ = np.max(epi)
           if max_>0.1:
              hist,bin_ = np.histogram(epi,range=(0.1,max_),bins=nb_,density=True)
              plt.plot(bin_[:-1],hist,alpha=0.5,color='green',linestyle='--',
                       label=r"$\pi$ bond of %s" %bd)

           epp = np.reshape(Epp[bd],[-1])
           max_ = np.max(epp)
           if max_>0.1:
              hist,bin_ = np.histogram(epp,range=(0.1,max_),bins=nb_,density=True)
              plt.plot(bin_[:-1],hist,alpha=0.5,color='blue',linestyle='-.',
                       label=r"$\pi\pi$ bond of %s" %bd)


           plt.legend(loc='best')
           plt.savefig('ebond_%s.eps' %bd) 
           plt.close() 

           print('\n -- %s -- \n' %bd)
           for i in range(rn.nbd[bd]):
                if Esi[bd][i][0]>0.0:
                   print('-  %s rbd:' %bd,rbd[bd][i][0],
                         '-  %s Esi:' %bd,Esi[bd][i][0],
                         '-  %s Epi:' %bd,Epi[bd][i][0],
                         '-  %s Epp:' %bd,Epp[bd][i][0])



if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   
   pleb(direcs={'ch4':'/home/gfeng/siesta/train/ch4'})
