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
from irff.molecule import Molecules,enlarge # SuperCell,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol
from irff.md.gulp import opt

''' scale the crystal box, while keep the molecule structure unchanged
'''

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--i', default=0,type=int, help='index of atomic frame')
parser.add_argument('--n', default=8,type=int, help='ncpu')
args = parser.parse_args(sys.argv[1:])

lf = open('ffield.json','r')
j = js.load(lf)
lf.close()

A = opt(gen=args.g,step=1000,l=1,t=0.0000001,n=args.n, x=1,y=1,z=1)
# A = read(args.g,index=args.i)
A = press_mol(A)
x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)
# print(j['rcutBond'])
masses = np.sum(A.get_masses())
volume = A.get_volume()
density = masses/volume/0.602214129

m_  = Molecules(A,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,"others": 1.68},check=True)
nmol = len(m_)

ir = IRFF_NP(atoms=A,
             libfile='ffield.json',
             nn=True)

print('\nnumber of molecules:',nmol)
print('\ndensity of configuration:',density)
ff = [1.0,5.0] #,1.9 ,2.0,2.5,3.0,3.5,4.0
cell = A.get_cell()
e  = []
eg = []
 
for i,f in enumerate(ff):
    m = copy.deepcopy(m_)
    _,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
    ir.calculate(A)
    e.append(ir.E)

    A = opt(atoms=A,step=500,l=0,t=0.0000001,n=args.n, x=1,y=1,z=1)
    system('mv md.traj md_{:d}.traj'.format(i))
    e_ = A.get_potential_energy()
    eg.append(e_)
 

print('The binding energy: ',(eg[-1]-eg[0]),'average:', (eg[-1]-eg[0])/nmol)

