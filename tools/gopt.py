#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import argparse
from ase.io import read
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from irff.md.gulp import opt


help_ = 'run with commond: ./gopt.py --g=nmc.gen --i=10 --s=30 --l=1'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--g',default='poscar.gen',type=str, help='training steps')
parser.add_argument('--f',default=0,type=int, help='read the frame')
parser.add_argument('--l',default=0,type=int, help='optimize cell')
parser.add_argument('--i',default=10,type=int, help='optimize cell interval')
parser.add_argument('--x',default=1,type=int, help='super cell in x')
parser.add_argument('--y',default=1,type=int, help='super cell in y')
parser.add_argument('--z',default=1,type=int, help='super cell in z')
parser.add_argument('--p',default=0.0,type=float, help='pressure')
parser.add_argument('--step',default=10,type=int, help='the trajectory frame')
args    = parser.parse_args(sys.argv[1:])

atoms = read(args.g,index=args.f) 
his = TrajectoryWriter('opt.traj', mode='w')

for i in range(args.step):
    e,atoms = opt(atoms=atoms,step=args.i,l=args.l,lib='reaxff_nn',pressure=args.p)
    calc = SinglePointCalculator(atoms, energy=e)
    atoms.set_calculator(calc)
    his.write(atoms=atoms)

his.close()

