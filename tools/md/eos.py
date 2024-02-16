#!/usr/bin/env python
# coding: utf-8
import numpy as np
import copy
from ase.io import read
from ase.io.trajectory import TrajectoryWriter #,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.molecule import Molecules,enlarge # SuperCell,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP

A = read('hmxc.gen')
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)

m_  = Molecules(A)
nmol = len(m_)

ir = IRFF_NP(atoms=A,
             libfile='ffield.json',
             nn=True)

print('\nnumber of molecules:',nmol)

ff = [0.92,0.94,0.96,0.98,1.0,1.02,1.04,1.06,1.08,1.1,1.12]
cell = A.get_cell()

with TrajectoryWriter('md.traj',mode='w') as his:
    for f in ff:
        m = copy.deepcopy(m_)
        _,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
        ir.calculate(A)
        A.calc = SinglePointCalculator(A,energy=ir.E)
        his.write(atoms=A)
 