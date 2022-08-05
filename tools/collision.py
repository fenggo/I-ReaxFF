#!/usr/bin/env python
# coding: utf-8
import sys
import argh
import argparse
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.constraints import FixAtoms
from irff.irff import IRFF
from irff.AtomDance import AtomDance
from irff.md.irmd import IRMD


'''
usage:
   ./collision.py --i=2 --j=7 --g=md.traj --T=2000
   use ./collision.py --h to see all options
'''

parser = argparse.ArgumentParser(description='Molecular Collision')
parser.add_argument('--g',default='poscar.gen',type=str, help='atomic configuration file name')
parser.add_argument('--i',default=0,type=int, help='index of atom i')
parser.add_argument('--j',default=1,type=int, help='index of atom j')
parser.add_argument('--f',default=-1,type=int, help='index of atom trajectory')
parser.add_argument('--step',default=100,type=int, help='time step of MD simulation')
parser.add_argument('--direction',default=1,type=int, help='Collision direction, 1 or -1')
parser.add_argument('--T',default=2000.0,type=float, help='Temperature of the simulation system')
args = parser.parse_args(sys.argv[1:])

atoms = read(args.g,index=-1)
ad = AtomDance(atoms=atoms)

atoms,gi,gj = ad.set_bond_momenta(args.i,args.j,atoms,sign=args.direction)

irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=args.step,Tmax=10000,
             learnpair=(args.i,args.j),beta=0.8,groupi=gi,groupj=gj,
             ro=0.8,rmin=0.5,initT=args.T)
irmd.run()

irmd.close()
ad.close()

