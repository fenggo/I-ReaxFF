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
parser.add_argument('--x', default=0,type=int, help='whether scale in this direction')
parser.add_argument('--y', default=0,type=int, help='whether scale in this direction')
parser.add_argument('--z', default=0,type=int, help='whether scale in this direction')
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

cell = A.get_cell()


with TrajectoryWriter('md.traj',mode='w') as his:
    for f in ff:
        if args.x:
           x_[:,0] = x_[:,0]*f
           cell[0] = cell[0]*f
        if args.y:
           x_[:,1] = x_[:,1]*f
           cell[1] = cell[1]*f
        if args.z:
           x_[:,2] = x_[:,2]*f
           cell[2] = cell[2]*f
        A.set_positions(x_)
        A.set_cell(cell)
        ir.calculate(A)
        A.calc = SinglePointCalculator(A,energy=ir.E)
        his.write(atoms=A)
 
