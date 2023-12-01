#!/usr/bin/env python
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units,Atoms
from ase.visualize import view
from irff.irff_np import IRFF_NP
from irff.AtomDance import AtomDance
# from irff.tools import deb_energy
# from irff.deb.deb_bde import deb_energy #,deb_bo # ,deb_vdw


def deb_eover(images,i=0,j=1,figsize=(10,8),show=False,print_=True):
    bopsi,boppi,boppp,bo0,bo1,eb = [],[],[],[],[],[]
    eo,eu,el,esi,r = [],[],[],[],[]
    eo_,eu_,eb_ = [],[],[]
    
    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nn=True)
    ir.calculate_Delta(images[0])
    
    for i_,atoms in enumerate(images):       
        ir.calculate(atoms)      
        bo0.append(ir.bop[i][j])      
        bo1.append(ir.bo0[i][j])  
        eb.append(ir.ebond[i][j])    
        eo.append(ir.eover[i]+ir.eover[j])      
        eu.append(ir.eunder[i]+ir.eunder[j])
        r.append(ir.r[i][j])
        
        if print_:
           print('r: {:6.4f} bo: {:6.4f} eov: {:6.4f} eov_i: {:6.4f} eov_j: {:6.4f}'.format(ir.r[i][j],
                 ir.bo0[i][j],ir.eover[i]+ir.eover[j],ir.eover[i],ir.eover[j]))
    
    
    plt.figure(figsize=figsize)     
    # plt.plot(r,bo0,alpha=0.8,linewidth=2,linestyle='-',color='g',label=r'$BO^{t=0}$')
    # plt.plot(r,bo1,alpha=0.8,linewidth=2,linestyle='-',color='y',label=r'$BO^{t=1}$')
    #plt.plot(r,bo1,alpha=0.8,linewidth=2,linestyle='-',color='r',label=r'$E_{bond}$')
    plt.plot(r,eo,alpha=0.8,linewidth=2,linestyle='-',color='indigo',label=r'$E_{over}$')
    #plt.plot(r,eu,alpha=0.8,linewidth=2,linestyle='-',color='b',label=r'$E_{under}$ ')
    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('deb_bo.pdf')
    if show: plt.show()
    plt.close()


parser = argparse.ArgumentParser(description='plot energies')
parser.add_argument('--traj',default='md.traj',type=str,help='the trajectory name')
parser.add_argument('--i',default=0,type=int,help='id of atom i')
parser.add_argument('--j',default=1,type=int,help='id of atom j')
parser.add_argument('--s',default=1,type=int,help='show the figure, if False, then save figure to pdf')
args = parser.parse_args(sys.argv[1:])

images = Trajectory(args.traj)

deb_eover(images,i=args.i,j=args.j,show=args.s,print_=True)

