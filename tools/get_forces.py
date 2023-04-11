#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from irff.irff import IRFF
from ase.io import read,write
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units
from irff.md.gulp import get_gulp_forces

images = Trajectory('gulp.traj')
get_gulp_forces(images)

#images = Trajectory('md.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# print('\n autograde: \n')
# for f in forces:
#     print(f)

# print('\n gulp: \n')
# images = Trajectory('gulp_force.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# for f in forces:
#     print(f)

