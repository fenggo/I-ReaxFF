#!/usr/bin/env python
import argparse
from ase.io import read # ,write
#from ase.io.trajectory import Trajectory
from ase import Atoms
#import matplotlib.pyplot as plt
import numpy as np
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol
import sys
import argparse


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''

   parser = argparse.ArgumentParser(description='stretch molecules')
   parser.add_argument('--g', default='gulp.traj',type=str, help='trajectory file')
   args = parser.parse_args(sys.argv[1:])
   atoms = read(args.g)
   atoms = press_mol(atoms)
 
   # ir = IRFF_NP(atoms=atoms,
   #            libfile='ffield.json',
   #            nn=True)

   # ir.calculate(atoms)
   # print(ir.E)
   order = ['C','O','N','H']

   spes  = []
   pos   = []

   cell  = atoms.get_cell()
   position = atoms.get_positions()

   spec  = atoms.get_chemical_symbols()

   for sp in order:
       for i,s in enumerate(spec):
           if s==sp:
              spes.append(s)
              pos.append(position[i])

   A = Atoms(spes,pos,cell=cell,pbc=[True,True,True])
   A.write('POSCAR')


