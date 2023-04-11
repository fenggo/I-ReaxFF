#!/usr/bin/env python
# coding: utf-8
import sys
# import argh
import argparse
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance,get_group


help_ = 'run with commond: ./stretch.py --g=nm.gen --i=0 --j=1 --s=1.4 --e=1.9'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--g',default='poscar.gen',type=str, help='training steps')
parser.add_argument('--i',default=0,type=int, help='iatom')
parser.add_argument('--j',default=1,type=int, help='jatom')
parser.add_argument('--s',default=1.4,type=float, help='start radius')
parser.add_argument('--e',default=1.9,type=float, help='end radius')
parser.add_argument('--f',default=-1,type=int, help='the trajectory frame')

args    = parser.parse_args(sys.argv[1:])


atoms   = read(args.g,index=args.f)
ad      = AtomDance(atoms=atoms)
images  = ad.stretch([args.i,args.j],nbin=50,rst=args.s,red=args.e,scale=1.26,traj='md.traj')
ad.close()


