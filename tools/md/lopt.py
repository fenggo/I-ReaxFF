#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
from ase import Atom, Atoms
from ase.io import read
# from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.constraints import StrainFilter
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory


os.environ['ASE_LAMMPSRUN_COMMAND'] = 'mpirun -n 4 lammps'

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--g',default='md.traj', help='atomic configuration')
parser.add_argument('--i',default=-1,type=int, help='the index in atomic configurations')
parser.add_argument('--x',default=1,type=int, help='the supercell size in x')
parser.add_argument('--y',default=1,type=int, help='the supercell size in y')
parser.add_argument('--z',default=1,type=int, help='the supercell size in z')
args = parser.parse_args(sys.argv[1:])

# cmds = ['pair_style    quip',
#         'pair_coeff * * ../Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\" 6']
# parameters = {'pair_style': 'quip',
#               'pair_coeff': [' * * Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\" 6']}

files = ['Carbon_GAP_20_potential/Carbon_GAP_20.xml',
         'Carbon_GAP_20_potential/Carbon_GAP_20.xml.sparseX.GAP_2020_4_27_60_2_50_5_4361',
         'Carbon_GAP_20_potential/Carbon_GAP_20.xml.sparseX.GAP_2020_4_27_60_2_50_5_4362',
         'Carbon_GAP_20_potential/Carbon_GAP_20.xml.sparseX.GAP_2020_4_27_60_2_50_5_4363']

### super cell defination
G = read(args.g)*(args.x,args.y,args.z)

#lammps= LAMMPSlib(lmpcmds=cmds, log_file='graphene.log')
lammps = LAMMPS(files=files)
lammps.set(pair_style='quip')
lammps.set(pair_coeff=['* * Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\" 6'])
lammps.set(tmp_dir='./')
lammps.set(keep_alive=False)
# lammps.set(keep_tmp_files=True)
G.calc = lammps
print("Energy ", G.get_potential_energy())


sf  = StrainFilter(G)
opt = BFGS(sf)

traj = Trajectory('path.traj', 'w', G)
opt.attach(traj)

opt.run(0.00001)

traj = Trajectory('path.traj')
atoms = traj[-1]
traj.close()

if args.x>1 or args.y>1 or args.z>1:   # get unit cell
   ncell     = args.x*args.y*args.z
   natoms    = int(len(atoms)/ncell)
   species   = atoms.get_chemical_symbols()
   positions = atoms.get_positions()
   forces    = atoms.get_forces()
   cell      = atoms.get_cell()
   cell      = [cell[0]/args.x, cell[1]/args.y,cell[2]/args.z]
   u         = np.linalg.inv(cell)
   pos_      = np.dot(positions[0:natoms], u)
   posf      = np.mod(pos_, 1.0)          # aplling simple pbc conditions
   pos       = np.dot(posf, cell)
   atoms     = Atoms(species[0:natoms],pos,#forces=forces[0:natoms],
                     cell=cell,pbc=[True,True,True])

atoms.write('POSCAR.unitcell')
