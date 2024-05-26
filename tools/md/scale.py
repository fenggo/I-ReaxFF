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

''' scale the crystal box, while keep the molecule structure unchanged
'''

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--i', default=0,type=int, help='trajectory index')
args = parser.parse_args(sys.argv[1:])

lf = open('ffield.json','r')
j = js.load(lf)
lf.close()

A = read(args.g,index=args.i)
# A = press_mol(A)
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)

ir = IRFF_NP(atoms=A,
             libfile='ffield.json',
             nn=True)

ff = [0.94,0.95,0.96,0.97,0.98,0.99,1.0,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1]
ff = [0.99,0.98,0.97]
cell = A.get_cell()

with TrajectoryWriter('md.traj',mode='w') as his:
    for f in ff:
        A.set_positions(x_*f)
        A.set_cell(cell*f)
        ir.calculate(A)
        A.calc = SinglePointCalculator(A,energy=ir.E)
        his.write(atoms=A)
 
