#!/usr/bin/env python
from os import getcwd,listdir
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.irff_np import IRFF_NP
# from irff.molecule import press_mol

cdir    = getcwd()
files   = listdir(cdir)
traj    = TrajectoryWriter('md.traj',mode='w')
poscars = []

for fil in files:
    f = fil.split('.')
    if len(f)>=1:
       if f[0]=='POSCAR':
          poscars.append(fil)

atoms = read(poscars[0])
ir    = IRFF_NP(atoms=atoms,libfile='ffield.json',nn=True)

for p in poscars:
    atoms = read(p)
    ir.calculate(atoms)
    atoms.calc = SinglePointCalculator(atoms,energy=ir.E)
    traj.write(atoms=atoms)
 
traj.close()
    