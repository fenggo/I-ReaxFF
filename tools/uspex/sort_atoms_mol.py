#!/usr/bin/env python
import argparse
import numpy as np
from ase.io import read # ,write
#from ase.io.trajectory import Trajectory
from ase import Atoms
#import matplotlib.pyplot as plt
from irff.molecule import Molecules,SuperCell
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol
import sys
import argparse

parser = argparse.ArgumentParser(description='stretch molecules')
parser.add_argument('--g', default='gulp.traj',type=str, help='trajectory file')
parser.add_argument('--x',default=1,type=int, help='repeat structure in x direction')
parser.add_argument('--y',default=1,type=int, help='repeat structure in y direction')
parser.add_argument('--z',default=1,type=int, help='repeat structure in z direction')
args = parser.parse_args(sys.argv[1:])

ii         = args.z
jj         = args.y
kk         = args.x
atoms      = read(args.g)  # *(args.x,args.y,args.z)
atoms      = press_mol(atoms)
cell       = atoms.get_cell()
natom      = len(atoms)
positions  = atoms.get_positions()
elems_     = atoms.get_chemical_symbols()
a = cell[0]
b = cell[1]
c = cell[2]

n     = natom*args.x*args.y*args.z
elems = [0 for i in range(n)]
x     = np.zeros([n,3])

for i in range(ii):
    for j in range(jj):
        for k in range(kk):
            for n in range(natom):
                print(i*jj*kk*natom+j*kk*natom+k*natom+n)
                print(i,j,k,n)
                elems[i*jj*kk*natom+j*kk*natom+k*natom+n] = elems_[n]   
                x[i*jj*kk*natom+j*kk*natom+k*natom+n][0]  = positions[n][0] + k*a[0] + j*b[0] +i*c[0]
                x[i*jj*kk*natom+j*kk*natom+k*natom+n][1]  = positions[n][1] + k*a[1] + j*b[1] +i*c[1]
                x[i*jj*kk*natom+j*kk*natom+k*natom+n][2]  = positions[n][2] + k*a[2] + j*b[2] +i*c[2]

a = [r*args.x for r in a]
b = [r*args.y for r in b]
c = [r*args.z for r in c]

atoms = Atoms(elems, x)
atoms.set_cell([a,b,c])
atoms.set_pbc([True,True,True])

positions  = atoms.get_positions()
elems      = atoms.get_chemical_symbols()
cell       = atoms.get_cell()
natom      = len(atoms)
spes       = []
pos        = []
mols       = []
order      = ['C','O','N','H'] 

m_     = Molecules(atoms,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,
                               "others": 1.68},
                   species=order,
                   check=True)
# m_,atoms = SuperCell(m_,cell=cell,supercell=[args.x,args.y,args.z])
# cell = atoms.get_cell()
nmol   = len(m_)
print('CY: {:5s}  NM: {:4d}'.format(args.g.split('.')[1],nmol))

for m in m_:
    # print(m.label)
    # print(m.mol_index)
    m.mol_index.sort()
    # print(m.mol_index)
    if m.label not in mols:
       mols.append(m.label) 

for sp in order:
    for mol in mols:
        for m in m_:
            if m.label==mol:
               for s in m.mol_index:
               # for s in range(natom):
                   if elems[s]==sp: #and s in m.mol_index:
                      spes.append(sp)
                      pos.append(positions[s])

A = Atoms(spes,pos,cell=cell,pbc=[True,True,True])
A.write('POSCAR')
