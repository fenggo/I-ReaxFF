#!/usr/bin/env python
import sys
import argparse
import numpy as np
# from ase import Atoms
# from ase.io import read, write
from ase.io.trajectory import Trajectory


parser = argparse.ArgumentParser(description='./train_torch.py --e=1000')
parser.add_argument('--t',default='md.traj' ,type=str, help='trajectory name')
args = parser.parse_args(sys.argv[1:])

images = Trajectory(args.t)

cfg = open('train.cfg','w')

for atoms in images:
    natom = len(atoms)
    energy= atoms.get_potential_energy()
    force = atoms.get_forces()
    cell  = atoms.get_cell()
    # print(energy)
    print('BEGIN_CFG',file=cfg)
    print(' Size',file=cfg)
    print('  {:d}'.format(natom),file=cfg)
    print(' Supercell',file=cfg)

    for c in cell:
        print('  {:f} {:f} {:f}'.format(c[0],c[1],c[2]),file=cfg)
    print(' AtomData: id type  cartes_x  cartes_y  cartes_z   fx   fy  fz',file=cfg)
    
    for i,atom in enumerate(atoms):
        print('{:5d}  0 {:10.7} {:10.7} {:10.7}  {:10.7}  {:10.7}  {:10.7}'.format(i+1,
                        atom.x,atom.y,atom.z,
                        force[i][0],force[i][1],force[i][2]),file=cfg)
    print(' Energy',file=cfg)
    print('  {:f}'.format(energy),file=cfg)
    print('END_CFG',file=cfg)
cfg.close()

