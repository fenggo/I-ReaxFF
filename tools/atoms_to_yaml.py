#!/usr/bin/env python
import sys
import argparse
import numpy as np
from ase.io import read
#from pymatgen.core import Structure
#from pymatgen.io.ase import AseAtomsAdaptor

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--gen',default='md.traj', help='atomic configuration')
parser.add_argument('--i',default=-1,type=int, help='the index in atomic configurations')
args = parser.parse_args(sys.argv[1:])

atoms = read(args.gen,index=args.i)
# atoms.write('poscar.yaml')

cell = atoms.get_cell()
cell = cell[:].astype(dtype=np.float32)
rcell     = np.linalg.inv(cell).astype(dtype=np.float32)
positions = atoms.get_positions()
xf        = np.dot(positions,rcell)
xf        = np.mod(xf,1.0)
fy   = args.gen[:-4] + '.yaml'

with open(fy,'w') as f:
     print('unit_cell:',file=f)
     print('  lattice:',file=f)
     print('    - [{:f},{:f},{:f}] '.format(cell[0][0],cell[0][1],cell[0][2],),file=f)
     print('    - [{:f},{:f},{:f}] '.format(cell[1][0],cell[1][1],cell[1][2],),file=f)
     print('    - [{:f},{:f},{:f}] '.format(cell[2][0],cell[2][1],cell[2][2],),file=f)
     print('  points:',file=f)
     symbols = atoms.get_chemical_symbols()
     for i,x in enumerate(xf):
         print('    - symbol: {:s}'.format(symbols[i]),file=f)
         print('      coordinates: [{:f},{:f},{:f}] '.format(x[0],x[1],x[2]),file=f)
