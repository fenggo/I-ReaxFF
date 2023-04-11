#!/usr/bin/env python
# coding: utf-8
import sys
# import argh
import argparse
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.irff import IRFF
from irff.AtomDance import AtomDance
from irff.md.irmd import IRMD


parser = argparse.ArgumentParser(description='Create a momentum between a pair atoms')
parser.add_argument('--g',default='poscar.gen',type=str, help='geomentry file')
parser.add_argument('--i',default=0,type=int, help='i atom')
parser.add_argument('--j',default=1,type=int, help='j atom')
parser.add_argument('--d',default=1,type=int, help='direction')
parser.add_argument('--s',default=100,type=int, help='md step')
parser.add_argument('--T',default=2000,type=float, help='moment temperature')
parser.add_argument('--b',default=0.96,type=float,help='the relax parameter')
parser.add_argument('--f',default=[],type=list,help='free atoms')
args    = parser.parse_args(sys.argv[1:])

atoms = read(args.g,index=-1)
ad = AtomDance(atoms=atoms)
# pairs = [[1,2],[13,7],[5,26]]
# images = ad.stretch([3,2],nbin=30,st=0.85,ed=1.25,scale=1.2,traj='md.traj')
ad.set_bond_momenta(args.i,args.j,atoms,sign=args.d)
ad.close()

f_= args.f if args.f else None

irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=args.s,Tmax=10000,
             freeatoms=f_,beta=args.b,
             ro=0.75,initT=args.T)
irmd.run()
mdsteps= irmd.step
Emd  = irmd.Epot
irmd.close()

''' run commond: ./moment.py --i=0 --j=1 --g=poscar.gen --T=2000 '''
