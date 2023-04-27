#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import argparse
from irff.irff_np import IRFF_NP


def nomb(traj='md.traj',interval=5):
    images = Trajectory(traj)

    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nomb=True,
                 nn=True)
    traj_ = TrajectoryWriter('nomb_'+traj,mode='w')

    for i,atoms in enumerate(images):
        if i%interval==0:
           ir.calculate(atoms)
           atoms.calc = SinglePointCalculator(atoms,energy=ir.E)
           traj_.write(atoms=atoms)
    traj_.close()

if __name__ == '__main__':
   ''' use commond like ./nomb.py <opt> to run it
       use --h to see options
   '''
   import sys
   import argparse

   help_ = './nomb.py --t=md.traj '
   parser = argparse.ArgumentParser(description=help_)
   parser.add_argument('--t',default='md.traj',type=str, help='atomic configuration')
   parser.add_argument('--i',default=5,type=int, help='time interval')
   args = parser.parse_args(sys.argv[1:])

   nomb(traj=args.t,interval=args.i)
