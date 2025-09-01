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

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--n',default=1,type=int, help='the number of cpu used in this calculation')
parser.add_argument('--x',default=1,type=int, help='X')
parser.add_argument('--y',default=1,type=int, help='Y')
parser.add_argument('--z',default=1,type=int, help='Z')
parser.add_argument('--g',default='POSCAR.unitcell',type=str, help='geometry file')
parser.add_argument('--step',default=500,type=int, help='Time Step')
args = parser.parse_args(sys.argv[1:])

os.environ['ASE_LAMMPSRUN_COMMAND'] = 'mpirun -n {:d} lammps'.format(args.n)
# cmds = ['pair_style    quip',
#         'pair_coeff * * ../Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\" 6']
# parameters = {'pair_style': 'quip',
#               'pair_coeff': [' * * Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\" 6']}
files = ['pot.almtp']
G = read(args.g)*(args.x,args.y,args.z)

#lammps= LAMMPSlib(lmpcmds=cmds, log_file='graphene.log')
lammps = LAMMPS(files=files)
lammps.set(pair_style='mlip load_from=pot.almtp')
lammps.set(pair_coeff=['* * # C'])
lammps.set(tmp_dir='./')
lammps.set(keep_tmp_files=False)
lammps.set(keep_alive=False)
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

if x>1 or y>1 or z>1:   # get unit cell
   ncell     = x*y*z
   natoms    = int(len(atoms)/ncell)
   species   = atoms.get_chemical_symbols()
   positions = atoms.get_positions()
   forces    = atoms.get_forces()
   cell      = atoms.get_cell()
   cell      = [cell[0]/x, cell[1]/y,cell[2]/z]
   u         = np.linalg.inv(cell)
   pos_      = np.dot(positions[0:natoms], u)
   posf      = np.mod(pos_, 1.0)          # aplling simple pbc conditions
   pos       = np.dot(posf, cell)
   atoms     = Atoms(species[0:natoms],pos,#forces=forces[0:natoms],
                     cell=cell,pbc=[True,True,True])

atoms.write('POSCAR.unitcell')
