#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance,get_group
import sys
import argh
import argparse

help_ = './stretch.py --g=md.traj --i=0 --j=1 --s=1.4 --e=1.9 --f=0'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--g',default='poscar.gen',type=str, help='training steps')
parser.add_argument('--i',default=0,type=int, help='learning rate')
parser.add_argument('--j',default=1,type=int, help='learning rate')
parser.add_argument('--s',default=1.4,type=float, help='learning rate')
parser.add_argument('--e',default=1.9,type=float, help='learning rate')
parser.add_argument('--f',default=-1,type=int, help='the trajectory frame')

args    = parser.parse_args(sys.argv[1:])


atoms   = read(args.g,index=args.f)
ad      = AtomDance(atoms=atoms)
images  = ad.stretch([args.i,args.j],nbin=50,rst=args.s,red=args.e,scale=1.26,traj='md.traj')
ad.close()


''' 
run this script using commond:
./stretch.py --g=md.traj --i=0 --j=1 --s=1.4 --e=1.9 --f=0
'''
