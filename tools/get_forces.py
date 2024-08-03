#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory,TrajectoryWriter
from irff.md.gulp import get_gulp_forces
from irff.irff_autograd import IRFF

images = Trajectory('md.traj')
atoms  = get_gulp_forces(images)
his    = TrajectoryWriter('auto_diff.traj',mode='w')
ir_    = IRFF(atoms=images[0],libfile='ffield.json',nn=True)

for img in images:
    ir_.calculate(atoms=img)
    forces = ir_.results['forces']
    img.calc = SinglePointCalculator(atoms, energy=ir_.E,forces=forces)
    his.write(atoms=img)

his.close()

#images = Trajectory('md.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# print('\n autograde: \n')
# for f in forces:
#     print(f)

# print('\n gulp: \n')
# # images = Trajectory('gulp_force.traj')
# # atoms  = images[0]
# forces = atoms.get_forces()
# for f in forces:
#     print(f)
