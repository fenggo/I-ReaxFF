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

structure.to(filename="POSCAR")

with open('POSCAR','r') as f:
     lines = f.readlines()

with open('POSCAR','w') as f:
     card = False
     for line in lines:
         if line.find('direct')>=0:
            card = True
         if card and line.find('direct')<0:
            print(line[:-3],file=f)
         else:
            print(line[:-1],file=f)
            