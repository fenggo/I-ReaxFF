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



colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','agua','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']


def plev():
    atoms = Atoms('H2',
              positions=[(0, 0, 0), (r, 0, 0)],
              cell=[10.0, 10.0, 10.0],
              pbc=[1, 1, 1])
    ao = AtomDance(atoms)
    pairs = [0,1]
    images = ao.stretch(pairs,nbin=50,st=0.6,ed=5.0,scale=1.25,traj='evdw.traj')
    ao.close()
    # view(images)
    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)

    ev_,r_ = [],[]
    for atoms in images:
        ir.calculate(atoms)
        r_.append(ir.r[0][1])
        ev_.append(ir.Evdw)


    fig, ax = plt.subplots() 
    plt.plot(r_,ev_,label=r'$Evdw$ VS $r$', color='blue', 
             linewidth=2, linestyle='-')

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Evdw.eps') 
    plt.close()


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       pb:   plot bo uncorrected 
       t:   train the whole net
   '''
   plev('f2.gen')

