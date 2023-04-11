#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
import sys
import argh
import argparse


help_ ='./bend.py --g=nm.gen --i=0 --j=1 --k=2 --a=30.0'

parser   = argparse.ArgumentParser(description=help_)
parser.add_argument('--g',default='poscar.gen',type=str, help='atomic configuration')
parser.add_argument('--i',default=0,type=int, help='atoms i of the angle')
parser.add_argument('--j',default=1,type=int, help='atoms j of the angle')
parser.add_argument('--k',default=2,type=int, help='atoms k of the angle')
parser.add_argument('--f',default=-1,type=int, help='atoms index of the trajectory')
parser.add_argument('--ar',default=30.0,type=float, help='angle range')

args     = parser.parse_args(sys.argv[1:])

atoms    = read(args.g,index=args.f)
# traj.write(atoms=atoms)

ad       = AtomDance(atoms)
images   = ad.swing_group([args.i,args.j,args.k],rang=args.ar,nbin=50,traj='md.traj')
# images = ad.bend([args.i,args.j,args.k],rang=args.ar,nbin=30,traj='md.traj')
# images = ad.rotate(atms=[0,1,2,3,4,5],axis=[6,7],o=6,rang=30.0,nbin=20,wtraj=True)
# ad.zmat_to_cartation(atoms,ad.InitZmat)
ad.close()
# view(atoms)




