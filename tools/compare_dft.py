#!/usr/bin/env python
# coding: utf-8
#from ase import Atoms
from ase.io.trajectory import Trajectory
from irff.plot.deb_bde import compare_dft_energy
import sys
import argparse

parser = argparse.ArgumentParser(description='plot energies')
parser.add_argument('--i',default=0,type=int,help='the id of atom i')
parser.add_argument('--j',default=1,type=int,help='the id of atom j')
parser.add_argument('--traj',default='md.traj',type=str,help='the dft ')
parser.add_argument('--csv',default=0,type=int,help='whether save the .csv data file')
parser.add_argument('--s',default=1,type=int,help='show the figure, if False, then save figure to pdf')
args = parser.parse_args(sys.argv[1:])


images = Trajectory(args.traj) 
#e = deb_energy(images,atomi=0,atomj=1,r_is_x=True,show=args.s,nn=True,figsize=(8,8),savecsv=args.csv)
e  = compare_dft_energy(images,atomi=args.i,atomj=args.j,show=args.s,nn=True)

''' Usage: ./compare_dft.py --t=md.traj --i=0 --j=1 '''

