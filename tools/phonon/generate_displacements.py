#!/usr/bin/env python
"""A script to generate supercells with displacements for LAMMPS."""
import numpy as np
from ase.io import read
import phonopy
from phonopy.interface.calculator import write_supercells_with_displacements
from phonopy.interface.phonopy_yaml import read_cell_yaml


atoms = read('POSCAR.unitcell')
# atoms.write('poscar.yaml')

cell = atoms.get_cell()
cell = cell[:].astype(dtype=np.float32)
rcell     = np.linalg.inv(cell).astype(dtype=np.float32)
positions = atoms.get_positions()
xf        = np.dot(positions,rcell)
xf        = np.mod(xf,1.0)
fy        =  'unitcell.yaml'

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

cell = read_cell_yaml("unitcell.yaml")
ph = phonopy.load(
    unitcell=cell,
    primitive_matrix="auto",
    supercell_matrix=[8, 8, 1],
    calculator="lammps",
)
ph.generate_displacements()
ph.save("phonopy_disp.yaml")
write_supercells_with_displacements(
    "lammps", ph.supercell, ph.supercells_with_displacements
)
