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

''' Use it like:
    ./dinfo.py --t=md.traj --i=5 --j=20
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
ir = IRFF_NP(atoms=A, libfile='ffield.json',nn=True)
ir.calculate(A)
print('\nTotal energy: \n',ir.E)

D = []
D_= []
for A in images:
    ir.calculate(A)
    positions = A.positions
    e = ir.E
    # print('\nEnergy: \n',e)
    D.append([ir.D_si[0][iatom]-ir.bop_si[iatom][jatom], 
              ir.D_pi[0][iatom]-ir.bop_pi[iatom][jatom], 
              ir.D_pp[0][iatom]-ir.bop_pp[iatom][jatom], 
              ir.bop[iatom][jatom], 
              ir.D_pp[0][jatom]-ir.bop_pp[iatom][jatom], 
              ir.D_pi[0][jatom]-ir.bop_pi[iatom][jatom], 
              ir.D_si[0][jatom]-ir.bop_si[iatom][jatom] ])
    D_.append([ir.Deltap[iatom]-ir.bop[iatom][jatom], ir.bop[iatom][jatom], 
               ir.Deltap[jatom]-ir.bop[iatom][jatom]])

print('\nD (7 element vector): \n')
for d in D:
    print('{:9.6f} {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:9.6f} {:9.6f}'.format(d[0],
            d[1],d[2],d[3],d[4],d[5],d[6]))
print('\nD (3 element vector): \n')
for d in D_:
    print('{:9.6f} {:9.6f} {:9.6f}'.format(d[0],d[1],d[2]))
