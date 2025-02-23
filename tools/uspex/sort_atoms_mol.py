#!/usr/bin/env python
import argparse
from ase.io import read # ,write
#from ase.io.trajectory import Trajectory
from ase import Atoms
#import matplotlib.pyplot as plt
from irff.molecule import Molecules,moltoatoms
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

atoms = read(args.g)*(args.x,args.y,args.z)
atoms = press_mol(atoms)
cell  = atoms.get_cell()

spes  = []
pos   = []
mols  = []
order = ['C','O','N','H'] 

m_     = Molecules(atoms,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,
                               "others": 1.68},
                   species=order,
                   check=True)
nmol   = len(m_)
print('CY: {:5s}  NM: {:4d}'.format(args.g.split('.')[1],nmol))

for m in m_:
    print(m.label)
    if m.label not in mols:
       mols.append(m.label) 

for sp in order:
    for mol in mols:
        for m in m_:
            atoms = moltoatoms([m])
            if m.label==mol:
                position = atoms.get_positions()
                spec     = atoms.get_chemical_symbols()
                for i,s in enumerate(spec):
                    if s==sp:
                        spes.append(s)
                        pos.append(position[i])

A = Atoms(spes,pos,cell=cell,pbc=[True,True,True])
A.write('POSCAR')
