#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from irff.dft.siesta import single_point
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff import IRFF
from irff.AtomDance import AtomDance


def angle_to_bond(angle=[1,0,2],xcf='VDW',xca='DRSLL',basistype='DZP',cpu=4):
    i,j,k = angle
    colors = ['r','b','k','y']
    markers= ['s','o','^','+']
    his = TrajectoryWriter('angle_to_bond.traj',mode='w')

    atoms   = read('h2o.gen')
    ad      = AtomDance(atoms)
    
    ang_lo  = 90
    ang_hi  = 140
    bin_    = 5
    ang_    = ang_lo

    Eir,Edft,R,angles = [],[],[],[]
    while ang_<ang_hi:
        ang_  += bin_
        e_,e,r    = [],[],[]
        img_   = ad.swing(angle,st=ang_lo,ed=ang_,nbin=10,wtraj=True)
        images = ad.stretch([i,j],atoms=img_[-1],nbin=15,st=0.8,ed=1.2,scale=1.26)

        for atoms in images:
            e.append(atoms.get_potential_energy()) # calculated by irff
            atoms_= single_point(atoms,xcf=xcf,xca=xca,basistype=basistype,cpu=cpu)
            e_.append(atoms_.get_potential_energy())
            his.write(atoms=atoms_)

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
        Eir.append(e)
        Edft.append(e_)
        angles.append(a)
        
    his.close()
    plt.figure()
    plt.ylabel('Energy (eV)')
    plt.xlabel('Radius (Angstrom)')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    for i_,e in enumerate(Eir):
        e_min = min(e)
        e = np.array(e) - e_min
        e_min = min(e_)
        e_= np.array(e_) - e_min

        plt.plot(R[i_],e,alpha=0.8,
                 linestyle='-',# marker='^',markersize=5,
                 color=colors[i_%4],label='H-O-H: %4.1f' %angles[i_])
        plt.plot(R[i_],e_,alpha=0.8,
                 linestyle=':',# marker='^',markersize=5,
                 color=colors[i_%4],label='H-O-H: %4.1f' %angles[i_])

    # plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('Energy.pdf',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   angle_to_bond()

