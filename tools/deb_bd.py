#!/usr/bin/env python
from __future__ import print_function
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import numpy as np
from irff.irff import IRFF
from irff.irff_np import IRFF_NP
from irff.data.ColData import ColData
from irff.dft.CheckEmol import check_emol
from irff.mpnn import MPNN
import matplotlib.pyplot as plt


ffield     = 'ffield.json'  
frame      = -1

images     = Trajectory('gulp.traj')
batch_size = (len(images))
atoms      = images[frame]

ir = IRFF_NP(atoms=atoms,
         libfile=ffield,
         nn=True)
ir.calculate(atoms)
natom   = ir.natom

ea      = ir.eang 
f7      = ir.f_7 
f8      = ir.f_8 
expang  = ir.expang 
theta   = ir.theta 
theta0  = ir.thet0 
sbo3    = ir.SBO3 
sbo     = ir.SBO 
pbo     = ir.pbo 
rnlp    = ir.rnlp 
# fa = open('ang.txt','w')
eb      = ir.bo0 
D       = ir.Deltap

print('\n bo: \n')
for i in range(natom):
    print('{:10d}'.format(i+1),end=' ')
print()
for i,eb_ in enumerate(eb):
    for e in eb_:
        print('{:10.6f}'.format(e),end=' ')
    print('{:4d}'.format(i+1))

bop      = ir.bop 

print('\n bop: \n')
for i in range(natom):
    print('{:10d}'.format(i+1),end=' ')
print()

for i,b_ in enumerate(bop):
    for b in b_:
        print('{:10.6f}'.format(b),end=' ')
    print('{:4d}'.format(i+1))

print('\n D: \n')
for i,d in enumerate(D):
    print('{:4d} {:10.6f}'.format(i+1,d-ir.P['val'][i]))



