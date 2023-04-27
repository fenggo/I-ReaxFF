#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import argh
import argparse
from irff.irff_np import IRFF_NP


def nomb(traj='md.traj'):
    images = Trajectory(traj)

    ir = IRFF_NP(atoms=images[0],
                 libfile='ffield.json',
                 nomb=True,
                 nn=True)
    traj_ = TrajectoryWriter('nomb_'+traj,mode='w')

    for atoms in images:
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
   args = parser.parse_args(sys.argv[1:])

   nomb(traj=args.t)
