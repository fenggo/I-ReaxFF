#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.io.espresso import read_espresso_in,parse_pwo_start,get_atomic_positions,label_to_symbol,get_constraint
from ase.visualize import view
from ase.units import create_units
from ase.dft.kpoints import kpoint_convert
from ase.calculators.singlepoint import SinglePointDFTCalculator,SinglePointKPoint
from ase.atoms import Atoms
from irff.dft.qe import read_espresso_out
import numpy as np


his = TrajectoryWriter('pw.traj',mode='w')

# atoms = read_espresso_in('2C.in')
images  = read_espresso_out('2B.out')
images_ = []
for a in images:
    his.write(atoms=a)
    images_.append(a)
    
his.close()
# view(images_)
