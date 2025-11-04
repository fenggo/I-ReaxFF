#!/usr/bin/env python
import subprocess
from os import getcwd,listdir,system
from os.path import exists
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.irff_np import IRFF_NP
# from irff.molecule import press_mol

density = 1.7
cdir    = getcwd()
cdir_   = '/'.join(cdir.split('/')[:-1])
# print('当前目录：',cdir_)
# dirs    = listdir(cdir_)
# for dir_ in dirs:
    #d = dir_.split('-')
#    if dir_.isalnum(): # d[1].isalpha(): isalnum()
if exists('{:s}/density.log'.format(cdir_)):
   with open('{:s}/density.log'.format(cdir_),'r') as f:
    for i,line in enumerate(f.readlines()):
        if i==0:
           continue
        l = line.split()
        den = float(l[1])
        # id_ = l[0]
        if den>=density:
           print(l[0],den)
           if exists('{:s}/{:s}/POSCAR.{:s}'.format(cdir_,l[0],l[0])):
              system('cp {:s}/{:s}/POSCAR.{:s} ./'.format(cdir_,l[0],l[0]))

#################### pack poscars to md.traj ######################
#cdir    = getcwd()
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
