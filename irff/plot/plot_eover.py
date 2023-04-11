#!/usr/bin/env python
# coding: utf-8
import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
from ase.visualize import view
from irff.irff_np import IRFF_NP
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from irff.AtomDance import AtomDance


def eover(i=0,j=1,ffield='ffield.json',nn='T',traj='md.traj'):
    # atoms = read(traj)
    # ao = AtomDance(atoms)
    # images = ao.stretch([[i,j]],nbin=50,traj=False)
    images =  Trajectory(traj)
    atoms=images[0]
    
    nn_=True if nn=='T'  else False
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn_)

    ir.calculate_Delta(atoms)
    natom = ir.natom

    r_,eb,bosi,bop_si,bop,bop_pi,bop_pp,bo = [],[],[],[],[],[],[],[]
    eba,eo,dlpi,dlpj,ev,boe = [],[],[],[],[],[]
    esi,epi,epp = [],[],[]
    Di,Dj = [],[]
    Dpi   = []
   
    for atoms in images:
        positions = atoms.positions
        v = positions[j] - positions[i]
        r = np.sqrt(np.sum(np.square(v)))
        
        ir.calculate(atoms)
        r_.append(ir.r[i][j])
        eb.append(ir.ebond[i][j])
        eba.append(ir.ebond[i][j] + ir.eover[i] + ir.Evdw) 
        ev.append(ir.Evdw)
        eo.append(ir.Eover) 
        # print(ir.so[j],ir.eover[j])

        dlpi.append(ir.Delta_lpcorr[i])
        dlpj.append(ir.Delta_lpcorr[j])
        Di.append(ir.Delta[i])
        Dj.append(ir.Delta[j])
        Dpi.append(ir.Dpil[j])

    fig, ax = plt.subplots() 
    ax.plot(eo,label=r'$E_{over}$',color='r', linewidth=2, linestyle='-')
    
#     fig, ax = plt.subplots(2,1,2) 
#     plt.plot(r_,dlpj,label=r'$\Delta_{lp}$(%s%d)' %(ir.atom_name[j],j), 
#              color='b', linewidth=2, linestyle='-') # Dpil
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Eover.pdf') 
    # plt.show()
    plt.close()


if __name__ == '__main__':
   eover()


