#!/usr/bin/env python
# coding: utf-8
import numpy as np
import copy
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.structures import structure
from irff.molecule import Molecules,enlarge # SuperCell,moltoatoms
#from irff.md.lammps import writeLammpsData


A = read('hmxc.gen')
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)

m_  = Molecules(A)
nmol = len(m_)

print('\nnumber of molecules:',nmol)

ff = [0.9,0.92,0.94,0.96,0.98,1.0,1.02,1.04,1.06,1.08,1.1]
# ff = [1.02]
cell = A.get_cell()

with TrajectoryWriter('md.traj',mode='w') as his:
    for f in ff:
        m = copy.deepcopy(m_)
        _,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
        # natom = len(A)
        # print('\nnumber of atoms:',natom)
        # nmol = len(m)
        # print('\nnumber of molecules in super-cell:',nmol)
        # A.write('tatb_super.gen')
        his.write(atoms=A)
    # view(A)
 
 
