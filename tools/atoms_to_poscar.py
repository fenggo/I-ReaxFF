#!/usr/bin/env python
import sys
import argparse
from ase.io import read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--gen',default='md.traj', help='atomic configuration')
parser.add_argument('--i',default=-1,type=int, help='the index in atomic configurations')
args = parser.parse_args(sys.argv[1:])

atoms = read(args.gen,index=args.i)
structure = AseAtomsAdaptor.get_structure(atoms)

cell = atoms.get_cell()
angles = cell.angles()
lengths = cell.lengths()

structure.to(filename="POSCAR")

with open('POSCAR','r') as f:
     lines = f.readlines()

with open('POSCAR','w') as f:
     card = False
     for i,line in enumerate(lines):
         if line.find('direct')>=0:
            card = True
         if card and line.find('direct')<0:
            print(line[:-3],file=f)
         elif i==0:
            print('EA {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f} Sym.group: 1'.format(lengths[0],
                       lengths[1],lengths[2],
                       angles[0],angles[1],angles[2]),file=f)
         else:
            print(line[:-1],file=f)
            