#!/usr/bin/env python
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import numpy as np
from irff.reax_force import ReaxFF_nn_force
from irff.irff_np import IRFF_NP
from irff.irff_autograd import IRFF
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
import matplotlib.pyplot as plt
from irff.md.gulp import get_gulp_forces


ir = ReaxFF_nn_force(dataset={'md':'md.traj'},
                     libfile='ffield.json')
ir.forward()

print('\n---- reax_nn_force ----\n')
for s in ir.bop:
    print(ir.E[s])


print('\n---- irff ----\n')
images = Trajectory('md.traj')
ir_ = IRFF(atoms=images[0],libfile='ffield.json',nn=True)
ir2 = IRFF_NP(atoms=images[0],libfile='ffield.json',nn=True)

forces = images[0].get_forces()


for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    ir2.calculate(atoms=img)
    print('--     IR     --      RForce     --     IRNP     --' )
    print(ir_.E,ir.E[s][i].item(),ir2.E)
    print(ir_.Eover.item(),ir.eover[s][i].item(),ir2.Eover)
    print(ir_.Eunder.item(),ir.eunder[s][i].item(),ir2.Eunder)
    print(ir_.Eang.item(),ir.eang[s][i].item(),ir2.Eang)
    # print('\n IR-dpi \n',ir2.Dpil)
 
 
print('\n----  forces  ----\n')
ir_.calculate(atoms=images[0])
for i in range(ir_.natom):
    print(ir_.results['forces'][i],'----' ,ir.force[s][0][i].detach().numpy(),
             '----',forces[i])

# get_gulp_forces(images)
# print('\n lammps: \n')
# images = Trajectory('md.traj')
# atoms  = images[0]
# forces = atoms.get_forces()
# for f in forces:
#     print(f)

