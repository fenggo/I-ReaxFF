#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
from ase import Atoms
from ase.io.trajectory import Trajectory
from irff.plot.deb_bde import deb_energy,deb_gulp_energy,compare_dft_energy,plot

desc_  = ''' 
      run with commond: ./deb_energies.py --e=HH --rmin=0.9 --rmax=1.6 
                    or  ./deb_energies.py --t=md.traj --i=0 --j=1  '''

parser = argparse.ArgumentParser(description=desc_)
parser.add_argument('--rmin',default=1.2,type=float,help='the minimus radius')
parser.add_argument('--rmax',default=2.0,type=float,help='the maximus radius')
parser.add_argument('--elements',default='CC',type=str,help='the element pair')
parser.add_argument('--traj',default=None,type=str,help='the trajectory name')
parser.add_argument('--csv',default=0,type=int,help='whether save the .csv data file')
parser.add_argument('--i',default=0,type=int,help='id of atom i')
parser.add_argument('--j',default=1,type=int,help='id of atom j')
parser.add_argument('--s',default=1,type=int,help='show the figure, if False, then save figure to pdf')
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

e = deb_energy(images,atomi=args.i,atomj=args.j,r_is_x=True,show=args.s,nn=True,figsize=(8,8),savecsv=args.csv)



