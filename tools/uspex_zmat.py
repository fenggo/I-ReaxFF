#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
from ase.io import read
from irff.AtomDance import AtomDance

parser = argparse.ArgumentParser(description='write the uspex style Z-matrix infomation')
parser.add_argument('--geo',default='POSCAR',type=str, help='geomentry file name')
parser.add_argument('--i',default=0,type=int, help='the i_th frame in traj file')
args = parser.parse_args(sys.argv[1:])

atoms  = read(args.geo,index=args.i)
ad     = AtomDance(atoms=atoms,rmax=1.2)
zmat   = ad.InitZmat
 
ad.write_zmat(zmat,uspex=True)
ad.close()

'''
run the script with command:
     ./uspex_zmat.py --g=structure.gen
'''

