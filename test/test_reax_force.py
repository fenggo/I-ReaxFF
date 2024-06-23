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


ir = ReaxFF_nn_force(dataset={'md':'md.traj'},
                     libfile='ffield.json')
ir.forward()

print('\n---- reax_nn_force ----\n')
for s in ir.bop:
    print(ir.E[s])


print('\n---- irff ----\n')
images = Trajectory('md.traj')
ir_ = IRFF(atoms=images[0],libfile='ffield.json',nn=True)

for i,img in enumerate(images):
    ir_.calculate(atoms=img)
    print(ir_.E,ir.E[s][i].item())
 
# for i,img in enumerate(images):
ir_.calculate(atoms=images[0])
print(ir_.results['forces'] ,ir.force[s][0])


