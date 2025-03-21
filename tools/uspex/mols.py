#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
import numpy as np
import copy
import json as js
from os import system
from ase.io import read
from ase.io.trajectory import TrajectoryWriter #,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.molecule import Molecules,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol
from irff.md.gulp import write_gulp_in,get_reax_energy
from irff.md.gulp import opt

''' scale the crystal box, while keep the molecule structure unchanged
'''

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--i', default=0,type=int, help='index of atomic frame')
parser.add_argument('--n', default=8,type=int, help='ncpu')
args = parser.parse_args(sys.argv[1:])


A = read(args.g,index=args.i)
m_  = Molecules(A,rcut={"H-O":1.12,"H-N":1.22,"H-C":1.22,"O-O":1.35,"others": 1.62},check=True)
nmol = len(m_)
print('\nnumber of molecules:',nmol)

cell = A.get_cell()
emolecules = 0.0

for i,m in enumerate(m_):
    atoms = moltoatoms([m])
    atoms.write('mol_{:d}.gen'.format(i))

 
