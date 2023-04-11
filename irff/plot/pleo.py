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


def pleo(atomi,traj='md.traj'):
    images = Trajectory(traj)
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)

    el_,eu_,eo_,r_ = [],[],[],[]
    delta = []
    for atoms in images:
        ir.calculate(atoms)
        # r_.append(ir.r[atomi][atomj])
        eo_.append(ir.eover[atomi])
        eu_.append(ir.eunder[atomi])
        el_.append(ir.elone[atomi])
        delta.append(ir.Delta[atomi])
        print('Delta_e:',ir.Delta_e[atomi],'Delta_lp:',ir.Delta_lp[atomi],
              'Delta_lpcorr:',ir.Delta_lpcorr[atomi])

    fig, ax = plt.subplots() 
    plt.plot(delta,eo_,label=r'$E_{over}$ VS $Radius$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Eover.pdf') 
    plt.close()

    fig, ax = plt.subplots() 
    plt.plot(delta,eo_,label=r'$E_{over}$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Eover.pdf') 
    plt.close()

    fig, ax = plt.subplots() 
    plt.plot(delta,eu_,label=r'$E_{under}$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Eunder.pdf') 
    plt.close()

    fig, ax = plt.subplots() 
    plt.plot(el_,label=r'$E_{lone}$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Elone.pdf') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   pleo(0)

