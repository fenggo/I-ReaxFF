#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
import sys
import argh
import argparse

''' 
run this script using commond:
./bend.py --g=nm.gen --i=0 --j=1 --k=2 --a=30.0 
'''

parser   = argparse.ArgumentParser(description='stretch molecules')
parser.add_argument('--g',default='poscar.gen',type=str, help='atomic configuration')
parser.add_argument('--atoms',default=[],type=list, help='atoms to rotate')
parser.add_argument('--x',default=[0,1],type=list, help='the axis to rotate')
parser.add_argument('--r',default=2,type=int, help='atoms range of the rotate angle')
parser.add_argument('--o',default=30.0,type=int, help='the suporting point of the rotate')

args     = parser.parse_args(sys.argv[1:])

atoms    = read(args.g,index=-1)
# traj.write(atoms=atoms)

ad       = AtomDance(atoms)
#images  = ad.swing_group([args.i,args.j,args.k],rang=args.ar,nbin=50,traj='md.traj')
# images = ad.bend([7,0,3],rang=25.0,nbin=30,traj='md.traj')
images   = ad.rotate(atms=args.atoms,axis=args.x,o=args.o,rang=args.r,nbin=50,traj='md.traj')
# ad.zmat_to_cartation(atoms,ad.InitZmat)
ad.close()
# view(atoms)


