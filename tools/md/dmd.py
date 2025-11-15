#!/usr/bin/env python
import sys
import argparse
from ase.io import read
from irff.dft.dftb import DFTB

help_ = './dmd.py --g=nm.gen  --s=100'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--gen',default='poscar.gen',type=str, help='gen file')
parser.add_argument('--i',default=-1,type=int, help='structure index in the trajectory file')
parser.add_argument('--step',default=300,type=int, help='step for geomentry optimization')
args    = parser.parse_args(sys.argv[1:])
gen_    = args.gen

if not gen_.endswith('.gen'):
   atoms   = read(args.gen,index=args.i)
   atoms.write('dftb.gen')
   gen_    = 'dftb.gen'

dftb = DFTB(maxscc=300,skf_dir='./')
dftb.opt(gen=gen_,latopt='yes',step=args.step)

