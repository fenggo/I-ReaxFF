#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import argparse
from irff.irff_np import IRFF_NP
from irff.md.gulp import nvt

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
   parser.add_argument('--i',default=1,type=int, help='time interval')
   args = parser.parse_args(sys.argv[1:])

   strucs = ['cl20','cl1','ch3nh2',
          'hmx','hmx1','hmx2',
          'fox','fox7',
          #'tnt',
          #'ch3no2','c2h4','c2h6','c3h8','c3h5',
          #'nmc',
          'h2o2','nm2','ch4w2',
          'nh3','n2h4','n22',
          'co2','no2']

   for st in strucs:
       nvt(gen=st+'.gen',T=350,time_step=0.1,tot_step=5000,movieFrequency=1)
       nomb(traj='md.traj',interval=args.i)
       print('save traj: nomb_{:s}.traj'.format(st))
       system('mv nomb_md.traj nomb_{:s}.traj'.format(st))

   # nomb(traj=args.t,interval=args.i)

