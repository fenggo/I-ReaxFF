#!/usr/bin/env python
import sys
# import argh
import argparse
from ase.io import read,write
# from ase import build
from irff.molecule import SuperCell

help_ = 'run with commond: ./supercell.py --g=nm.gen --x=1 --y=1 --z=2 '
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--g',default='poscar.gen',type=str, help='training steps')
parser.add_argument('--x',default=1,type=int, help='x supercell factor')
parser.add_argument('--y',default=1,type=int, help='y supercell factor')
parser.add_argument('--z',default=1,type=int, help='z supercell factor')
args    = parser.parse_args(sys.argv[1:])

A = read(args.g)
# build.make_supercell(A,[2,2,2])
# write('poscar.gen',A*(2,2,2))

_,atoms = SuperCell(A,fac=1.0,supercell=None)
write('POSCAR.supercell',atoms)
