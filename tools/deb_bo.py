#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
import numpy as np
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units,Atoms
from ase.visualize import view
from irff.irff_np import IRFF_NP
# from irff.tools import deb_energy
import matplotlib.pyplot as plt
from irff.plot.deb_bde import deb_bp,deb_bo # ,deb_vdw
# get_ipython().run_line_magic('matplotlib', 'inline')

parser = argparse.ArgumentParser(description='Plot bond-order and others')
parser.add_argument('--rmin',default=1.2,type=float,help='the minimus radius')
parser.add_argument('--rmax',default=2.0,type=float,help='the maximus radius')
parser.add_argument('--elements',default='CC',type=str,help='the element pair')
parser.add_argument('--traj',default=None,type=str,help='the trajectory name')
parser.add_argument('--i',default=0,type=int,help='id of atom i')
parser.add_argument('--j',default=1,type=int,help='id of atom j')
parser.add_argument('--s',default=1,type=int,help='show the figure, if False, then save figure to pdf')
parser.add_argument('--n',default=1,type=int,help='neural network bo correction')
parser.add_argument('--f',default='ffield.json',type=str,help='force field file')
args = parser.parse_args(sys.argv[1:])

r      = args.rmin
if args.traj is None:
   images = []
   while r<args.rmax:
      r += 0.1
      atoms = Atoms(args.elements,
                  positions=[(0, 0, 0), (r, 0, 0)],
                  cell=[10.0, 10.0, 10.0],
                  pbc=[1, 1, 1])
      images.append(atoms)
else:
   images = Trajectory(args.traj)

deb_bo(images,i=args.i,j=args.j,show=args.s,more=True,x_distance=True,print_=True,nn=args.n,libfile=args.f)

''' Usage: ./deb_bo.py --e=HH --rmin=0.9 --rmax=1.6 
           ./deb_bo.py --t=md.traj --i=0 --j=1  
'''
