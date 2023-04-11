#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff import IRFF


def e(i=1,j=0,k=2):
    colors = ['r','b','k','y']
    markers= ['s','o','^','+']
    images1 = Trajectory('h2o-s1.traj')
    images2 = Trajectory('h2o-s2.traj')
    images3 = Trajectory('h2o-s3.traj')
    images4 = Trajectory('h2o-s4.traj')

    E,R,angles = [],[],[]
    for images in [images1,images2,images3,images4]:
        e,r = [],[]
        for atoms in images:
            e.append(atoms.get_potential_energy())
            ri = atoms.positions[i]
            rj = atoms.positions[j]
            rk = atoms.positions[k]
            rij= rj - ri
            rjk= rk - rj
            rik= rk - ri
            rij2 = np.sum(np.square(rij))
            rij_ = np.sqrt(rij2)
            r.append(rij_)
            
        rik2 = np.sum(np.square(rik))
        rik_ = np.sqrt(rik2)
        rjk2 = np.sum(np.square(rjk))
        rjk_ = np.sqrt(rjk2)
        cos_ = (rij2+rjk2-rik2)/(2.0*rij_*rjk_)
        a    = np.arccos(cos_)*180.0/3.14159

        R.append(r)
        E.append(e)
        angles.append(a)

    plt.figure()
    plt.ylabel('Energy (eV)')
    plt.xlabel('Radius (Angstrom)')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    for i_,e in enumerate(E):
        e_min = min(e)
        e = np.array(e) - e_min

        plt.plot(R[i_],e,alpha=0.8,
                 linestyle='-',marker=markers[i_],markersize=5,
                 color='black',label='H-O-H: %4.1f' %angles[i_])

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Energy.pdf',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   e()

