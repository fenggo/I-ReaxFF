#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.AtomDance import AtomDance


atoms  = read('cl1.gen',index=0)
ad     = AtomDance(atoms=atoms,rmax=1.33)
ad.write_zmat(ad.InitZmat)
zmat   = ad.InitZmat
# zmat[2][1] = 109.0
# print(zmat)

images = []
his    = TrajectoryWriter('md.traj',mode='w')
ang    = 90.0

for j in range(40):
    ang += 1.0
    zmat[9][1] = ang
    atoms  = ad.zmat_to_cartation(atoms,zmat)
    ad.ir.calculate(atoms)
    atoms.calc = SinglePointCalculator(atoms,energy=ad.ir.E)
    his.write(atoms=atoms)
images.append(atoms.copy())

his.close()
ad.close()
view(images)



