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


def pleb(atomi=0,atomj=1,traj='md.traj'):
    images = Trajectory(traj)
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)

    e_,r_ = [],[]
    for atoms in images:
        ir.calculate(atoms)
        r_.append(ir.r[atomi][atomj])
        e_.append(ir.ebond[atomi][atomj])

    fig, ax = plt.subplots() 
    plt.plot(r_,e_,label=r'$E_{Bond}$ VS $Radius$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Ebond.eps') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   pleb(atomi=0,atomj=5)

