#!/usr/bin/env python
import sys
import argparse
from irff.dft.dftb import DFTB

help_ = './dmd.py --g=nm.gen  --s=100'
parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--g',default='poscar.gen',type=str, help='gen file')
parser.add_argument('--step',default=300,type=int, help='step for geomentry optimization')
args    = parser.parse_args(sys.argv[1:])

dftb = DFTB(maxscc=300,skf_dir='./')
dftb.opt(gen=args.gen,latopt='yes',step=args.step)

