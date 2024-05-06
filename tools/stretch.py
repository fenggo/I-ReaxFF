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
parser.add_argument('--auto',default=1,type=int, help='automaticly determin the moving group')
parser.add_argument('--f',default=-1,type=int, help='the trajectory frame')
parser.add_argument('--n',default=9,type=int, help='the number of trajectory frame')
args    = parser.parse_args(sys.argv[1:])


atoms   = read(args.g,index=args.f)
ad      = AtomDance(atoms=atoms,rmax=1.2,
                    rcut={"C-H":1.22,"H-O":1.22,"H-H":1.2,"O-O":1.4,
                          "others": 1.8})

if args.auto:
   to_move = None
else:
   to_move = [args.j]

images  = ad.stretch([args.i,args.j],nbin=args.n,rst=args.s,red=args.e,
                     ToBeMoved=to_move,traj='md.traj')
ad.close()


