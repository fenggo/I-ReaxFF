#!/usr/bin/env python
import numpy as np
import sys
import argparse
from ase.io.trajectory import Trajectory
from ase.io import read,write
from ase import units,Atoms
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from irff.irff_np import IRFF_NP
import matplotlib.pyplot as plt
from irff.plot.deb_bde import deb_energy,deb_bo


# 120*3.141593/180

def torsion(images,interval=5):
    atoms = images[0]
    ir = IRFF_NP(atoms=atoms,
                libfile='ffield.json',
                nn=True)
    ir.calculate(atoms)

    for t,tor in enumerate(ir.tors):
        i_,j_,k_,l_ = tor
        print('{:2d}: {:2d}-{:2d}-{:2d}-{:2d}  {:s}-{:s}-{:s}-{:s}  {:8.4f}'.format(t,i_,j_,k_,l_,
              ir.atom_name[i_],ir.atom_name[j_],ir.atom_name[k_],ir.atom_name[l_],ir.etor[t]))
    
    t = int(input('please input the id of the torsion angle to analysis: '))
    i,j,k,l = ir.tors[t]
    tor_ = []
    Etor   = []

    for i_,atoms in enumerate(images):
        if i_%interval!=0:
           continue
        ir.calculate(atoms)
        t = np.where(np.logical_and(np.logical_and(np.logical_and(ir.tors[:,0]==i,ir.tors[:,1]==j),
                                                   ir.tors[:,2]==k),
                                    ir.tors[:,3]==l))
        t = np.squeeze(t)
 
        print('{:3d}  boij: {:6.4f} bojk: {:6.4f} bokl: {:6.4f} '
              'Dj: {:6.4f} Dk: {:6.4f} f10: {:6.4f} f11: {:6.4f} fijkl: {:6.4f} '
              'cos: {:6.4f} Etor: {:8.6f}'.format(t,
              ir.botij[t],ir.botjk[t],ir.botkl[t],
              ir.Dang[ir.torj[t]],ir.Dang[ir.tork[t]],
              ir.f_10[t],ir.f_11[t],ir.fijkl[t],
              ir.cos_w[t],ir.etor[t])) 
        tor_.append(ir.cos_w[t])
        Etor.append(ir.etor[t])

    plt.figure()     
    plt.plot(tor_,Etor,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$Eangle$')
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.show() # if show else plt.savefig('deb_bo.pdf')
    plt.close()


help_ = 'run with commond: ./theta.py --t=md.traj'

parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--t',default='md.traj',type=str,help='the atomic gementry file name')
parser.add_argument('--interval',default=10,type=int,help='the time interval of the trajectory to compute')
args = parser.parse_args(sys.argv[1:])

images = Trajectory(args.t)
torsion(images,interval=args.interval)




