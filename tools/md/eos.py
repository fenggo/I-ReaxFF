#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
import numpy as np
import copy
import json as js
from ase.io import read
from ase.io.trajectory import TrajectoryWriter #,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.molecule import Molecules,enlarge # SuperCell,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
args = parser.parse_args(sys.argv[1:])

lf = open('ffield.json','r')
j = js.load(lf)
lf.close()

A = read(args.g)
A = press_mol(A)
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)

print(j['rcutBond'])

m_  = Molecules(A,rcut=j['rcutBond'])
nmol = len(m_)

ir = IRFF_NP(atoms=A,
             libfile='ffield.json',
             nn=True)

print('\nnumber of molecules:',nmol)

ff = [0.98,1.0,1.02,1.06,1.1]
# ff = [3]
cell = A.get_cell()

with TrajectoryWriter('md.traj',mode='w') as his:
    for f in ff:
        m = copy.deepcopy(m_)
        _,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
        ir.calculate(A)
        A.calc = SinglePointCalculator(A,energy=ir.E)
        his.write(atoms=A)
 