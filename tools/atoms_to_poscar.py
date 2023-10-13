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
