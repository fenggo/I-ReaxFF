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
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
import matplotlib.pyplot as plt


atoms = read('gp4.traj')

ir = ReaxFF_nn_force(dataset={'gp4':'gp4.traj'},
                     libfile='ffield.json')
ir.forward()

print('\n---- reax_nn_force ----\n')
for s in ir.bop:
    # print(ir.bo0[s])
    # print(ir.ebond[s])
    # print(ir.eang[s])
    # print(ir.epen[s])
    # print(ir.etor[s])
    # print(ir.efcon[s])
    # print(ir.evdw[s])
    # print(ir.ecoul[s])
    # print(ir.ehb[s])
    # print(ir.eself[s])
    # print(ir.zpe[s])
    print(ir.E[s])
    # print('\n-- bond list --\n',ir.bdid[s])
#     print(ir.bop[b].detach().numpy())
# print(ir.E)

print('\n---- irff ----\n')
ir_ = IRFF_NP(atoms=atoms,libfile='ffield.json',nn=True)
ir_.calculate(atoms=atoms)
# print(ir_.bo0)
# print(ir_.Ebond)
# print(ir_.Eang)
# print(ir_.Epen)
# print(ir_.Etor)
# print(ir_.Efcon)
# print(ir_.Evdw)
# print(ir_.Ecoul)
# print(ir_.Ehb)
# print(ir_.Eself)
# print(ir_.zpe)
print(ir_.E)
# print(ir_.E)
# print('\n F \n')


