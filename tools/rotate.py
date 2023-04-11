#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
import sys
import argh
import argparse


parser   = argparse.ArgumentParser(description='stretch molecules')
parser.add_argument('--g',default='poscar.gen',type=str, help='atomic configuration')
parser.add_argument('--i',default=0,type=int, help='atoms i of the angle')
parser.add_argument('--j',default=1,type=int, help='atoms j of the angle')
#parser.add_argument('--o',default=2,type=int, help='atoms k of the angle')
parser.add_argument('--f',default=-1,type=int, help='number of trajectory frame')
parser.add_argument('--d',default=90.0,type=float, help='angle rotate degree')

args     = parser.parse_args(sys.argv[1:])
atoms    = read(args.g,index=args.f)
ad       = AtomDance(atoms)
images = ad.rotate(axis=[args.i,args.j],o=args.j,rang=args.d,nbin=50,traj='md.traj')
ad.close()
# view(atoms)


''' 
run this script using commond:
./rotate.py --g=nm.gen --i=0 --j=1 --d=30.0 
'''

