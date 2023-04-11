#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.irff_np import IRFF_NP
from irff.AtomDance import AtomDance
import argh
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io.trajectory import Trajectory
from ase.io import read
import tensorflow as tf



colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','agua','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']


def ple(traj='md.traj',dE=0.15,d2E=0.05,Etole=0.1):
    images = Trajectory(traj)
    tframe = len(images)
    x_     = [i for i in range(tframe)]
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)

    energies,e_ = [],[]
    dEs,d2Es    = [],[]
    ind_,labels = [],[]
    dE_         = 0.0
    d2E_        = 0.0

    for i,atoms in enumerate(images):
        energy = atoms.get_potential_energy()
        ir.calculate(atoms)
        # e_.append(ir.E)

        if i>0:              ###########
           if i<(tframe-1):
              deltEl =  energy - energies[-1]
              deltEr =  images[i+1].get_potential_energy() - energy
              dE_ = abs(deltEl)
              d2E_= abs(deltEr-deltEl)
           else:
              deltEl =  energy - energies[-1]
              dE_ = abs(deltEl)

        energies.append(energy)
        dEs.append(dE_)
        d2Es.append(d2E_)
        labels.append(energy)
        print(' * differential:',dE_,d2E_)
        # if dE_>dE or d2E_>d2E:
        #    if i not in ind_:
        #       ind_.append(i)
        #       labels.append(energy)

    labels_ = []
    dft = False
    if isfile('SinglePointEnergies.log'):
       dft = True
       with open('SinglePointEnergies.log','r') as fs:
            in_,edft = [],[]
            for line in fs.readlines():
                l  = line.split()
                i_ = int(l[0])
                e  = float(l[2])
                e_ = float(l[4])
                dE = float(l[8])
                d2E= float(l[10])

                in_.append(i_)
                edft.append(e_)
                labels_.append(e)

    fig, ax = plt.subplots() 
    plt.ylabel('Energy (unit: eV)')
    plt.xlabel('Time (unit: fs)')
    le   = (len(energies)-1)*0.1
    dle  = le/100.0
    plt.xlim(-dle,le+dle)
    ind_ = np.array(ind_)
    emin = np.min(energies)
    labels = labels_

    i_,labels_ = [],[]
    for _,i in enumerate(in_):
        if abs(edft[_]-labels[_])>Etole:
           i_.append(i)
           labels_.append(labels[_])

    labels_ = np.array(labels_)
    i_      = np.array(i_)

    plt.scatter(i_*0.1,labels_-emin,color='none',edgecolors='red',linewidths=2,
                marker='*',s=150,label=r'$Labeled$ $Data$',
                alpha=1.0)

    if dft:
       err= np.array(edft) - np.array(labels)
       in_= np.array(in_)
       plt.errorbar(in_*0.1,edft-emin,yerr=err,fmt='s',ecolor='red',color='none',
                    ms=7,markerfacecolor='none',mec='red',
                    elinewidth=2,capsize=2,label=r'$True$ $Value(DFT)$')

    x_ = np.array(x_)
    plt.plot(x_*0.1,energies-emin,label=r'$IRFF-MPNN$ $Predictions$', color='blue', 
             marker='o',markerfacecolor='none',
             markeredgewidth=1, 
             ms=3,alpha=0.8,
             linewidth=1, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('popeng.svg') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   ple()


