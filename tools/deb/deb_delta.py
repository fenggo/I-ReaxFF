#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
from ase import Atoms
from ase.io import read
from irff.plot.deb_bde import deb_energy,deb_delta
from irff.md.gulp import get_gulp_forces

parser = argparse.ArgumentParser(description='print out delta')
parser.add_argument('--gen',default=None,type=str,help='the genmentry file name')
parser.add_argument('--f',default='ffield.json',type=str,help='force field file')
parser.add_argument('--n',default=1,type=int,help='neural network bo correction')
args = parser.parse_args(sys.argv[1:])


atoms = read(args.gen)
e = deb_delta(atoms,libfile=args.f,nn=args.n)

# get_gulp_forces(images,traj='gulp_force.traj',ffield='reaxff_nn.lib')

''' Usage: ./deb_delta.py --g=md.traj 
'''

