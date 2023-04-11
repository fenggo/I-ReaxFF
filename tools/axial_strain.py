#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from ase.io import read,write
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import TrajectoryWriter
from ase import units
from ase.visualize import view
from irff.md.gulp import opt


atoms = read('gulp.cif') 
# cell  = atoms.get_cell()[:]

#cell[2] = cell[2]*0.9996
#print(cell)
#atoms.set_cell(cell)
#view(atoms)
his = TrajectoryWriter('axial_strain.traj', mode='w')

for i in range(5):
    cell  = atoms.get_cell()[:]
    cell[2] = cell[2]*1.001 # 0.999
    atoms.set_cell(cell)
     
    e,atoms = opt(atoms=atoms,step=200,lib='reaxff_nn')
    calc = SinglePointCalculator(atoms, energy=e)
    atoms.set_calculator(calc)
    his.write(atoms=atoms)

his.close()

