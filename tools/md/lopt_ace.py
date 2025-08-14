#!/usr/bin/env python
import os
import numpy as np
from ase import Atom, Atoms
from ase.io import read
# from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from ase.constraints import StrainFilter
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory


os.environ['ASE_LAMMPSRUN_COMMAND'] = 'mpirun -n 4 lammps'


files = ['d2_short.table','c_ace.yace']

### super cell defination
x = 8   
y = 8
z = 1

G = read('POSCAR.unitcell')*(x,y,z)


#lammps= LAMMPSlib(lmpcmds=cmds, log_file='graphene.log')
lammps = LAMMPS(files=files)
lammps.set(pair_style='hybrid/overlay pace table linear 10000')
lammps.set(pair_coeff=['* * pace c_ace.yace C','* * table d2_short.table D2 9.0'])
lammps.set(tmp_dir='./')
# lammps.set(keep_tmp_files=True)
G.calc = lammps
print("Energy ", G.get_potential_energy())


sf  = StrainFilter(G)
opt = BFGS(sf)

traj = Trajectory('path.traj', 'w', G)
opt.attach(traj)

opt.run(0.000000001)

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
