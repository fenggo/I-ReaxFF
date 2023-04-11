#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
from irff.irff_np import IRFF_NP
from irff.AtomOP import AtomOP
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


def ple(traj='md.traj'):
    images = Trajectory(traj)
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)

    e,e_,r_ = [],[],[]
    for atoms in images:
    # for i in range(62):
    #    atoms = images[i]
        e.append(atoms.get_potential_energy())
        ir.calculate(atoms)
        # r_.append(ir.r[atomi][atomj])
        e_.append(ir.E)

    fig, ax = plt.subplots() 
    
    plt.plot(e,label=r'$DFT$ ($SIESTA$)', color='red', 
             markeredgewidth=1, 
             ms=5,alpha=0.8,
             linewidth=2, linestyle='-')

    # plt.plot(e,label=r'$Potential$ $Energy$', color='red', 
    #          marker='^',markerfacecolor='none',
    #          markeredgewidth=1, 
    #          ms=5,alpha=0.8,
    #          linewidth=1, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('popeng.svg',transparent=True) 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   ple()


