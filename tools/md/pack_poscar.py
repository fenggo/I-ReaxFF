#!/usr/bin/env python
import glob
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.irff_np import IRFF_NP
# from irff.molecule import press_mol

cdir    = getcwd()
poscars = glob.glob('*.gen')
traj    = TrajectoryWriter('md.traj',mode='w')
poscars = []

atoms = read(poscars[0])
ir    = IRFF_NP(atoms=atoms,libfile='ffield.json',nn=True)

for p in poscars:
    atoms = read(p)
    ir.calculate(atoms)
    atoms.calc = SinglePointCalculator(atoms,energy=ir.E)
    traj.write(atoms=atoms)
 
traj.close()
