#!/usr/bin/env python
import sys
import argparse
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.data import chemical_symbols

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--gen',default='md.traj', help='atomic configuration')
parser.add_argument('--i',default=-1,type=int, help='the index in atomic configurations')
args = parser.parse_args(sys.argv[1:])

atoms = read(args.gen,index=args.i)

masses = np.sum(atoms.get_masses())
volume = atoms.get_volume()
density = masses/volume/0.602214129

print('Volume: ',volume, 'Masses: ',masses,'Density: ',density)

