#!/usr/bin/env python
import numpy as np
from ase.io.trajectory import Trajectory
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
# from irff.molecule import press_mol

density = 1.4
inds = [i for i in range(71)]

########### pack to poscars ##########
images = Trajectory('Individuals.traj')

fposcars = open('POSCARS','a')
for i_,atoms in enumerate(images):
    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density_ = masses/volume/0.602214129
    if density_<density or i_ not in inds:
       continue
    
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="POSCAR")
    cell = atoms.get_cell()
    angles = cell.angles()
    lengths = cell.lengths()
    with open('POSCAR','r') as f:
         lines = f.readlines()
     
    card = False
    for i,line in enumerate(lines):
        if line.find('direct')>=0:
           card = True
        if card and line.find('direct')<0:
           print(line[:-3],file=fposcars)
        elif i==0:
           print('EA{:d} {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f} Sym.group: 1'.format(i_,
                   lengths[0],lengths[1],lengths[2],
                   angles[0],angles[1],angles[2]),file=fposcars)
        else:
           print(line[:-1],file=fposcars)

    # print('{:s}'.format(i_))
fposcars.close()

