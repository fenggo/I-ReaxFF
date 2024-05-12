#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
import numpy as np
import copy
from ase.io import read
#from ase.visualize import view
from ase.io.trajectory import TrajectoryWriter ,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.molecule import Molecules,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol

''' scale the crystal box, while keep the molecule structure unchanged
'''
#
#   parse argument
#
parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--t', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--iatom', default=0,type=int, help='i atom')
parser.add_argument('--jatom', default=1,type=int, help='j atom')
args = parser.parse_args(sys.argv[1:])

iatom = args.iatom
jatom = args.jatom

#
#   read in atoms structures, and recover molecules which 
#   is divided into fractions by box
#

images = Trajectory(args.t)
A = images[0]
A = press_mol(A)
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)

#
#  recongnize molecules
#
mols  = Molecules(A,rcut={"H-O":1.22,"H-H":1.2,"O-O":1.4,"others": 1.8},check=True)
nmol  = len(mols)
print('\nnumber of molecules in trajectory: {:d}'.format(nmol))

ir_total = IRFF_NP(atoms=A, libfile='ffield.json',nn=True)
ir_total.calculate(A)

print('\nTotal energy: \n',ir_total.E)


ir    = [None for i in range(nmol)] 
atoms = [None for i in range(nmol)] 

imol = 0
for i,m in enumerate(mols):
    atoms[i] = moltoatoms([m])
    ir[i] = IRFF_NP(atoms=atoms[i],libfile='ffield.json',nn=True)
    ir[i].calculate(atoms[i])
    # print('\nMolecular energy: \n',ir[i].E)
    # print(m.mol_index)
    # view(atoms)
    if iatom in m.mol_index and jatom in m.mol_index:
       imol = i

iatom_ = mols[imol].mol_index.index(iatom)
jatom_ = mols[imol].mol_index.index(jatom)

for A in images:
    ir_total.calculate(A)
    positions = A.positions
    # print('\nTotal energy: \n',ir_total.E)
    # print('\nMolecular energy: \n')
    e_other = ir_total.E
    for i,m in enumerate(mols):
        # print(m.mol_index)
        atoms[i].positions = positions[m.mol_index]
        ir[i].calculate(atoms[i])
        # print(ir[i].E,end=' ')
        e_other = e_other - ir[i].E
    print('\nothers: \n',e_other)


print('D ',ir_total.Deltap[iatom],ir_total.bo0[iatom][jatom],ir_total.Deltap[jatom])
print('D ',ir[imol].Deltap[iatom_],ir[imol].bo0[iatom_][jatom_],ir[imol].Deltap[jatom_])

