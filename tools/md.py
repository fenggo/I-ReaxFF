#!/usr/bin/env python
# coding: utf-8
import sys
from ase.optimize import BFGS,QuasiNewton
from ase.constraints import StrainFilter,FixAtoms
from ase.vibrations import Vibrations
from ase.io import read,write
from irff.irff import IRFF
from irff.AtomDance import AtomDance
from irff.md.irmd import IRMD
from ase.visualize import view
from ase.io.trajectory import Trajectory,TrajectoryWriter
import numpy as np
import argh
import argparse


parser = argparse.ArgumentParser(description='Molecular Dynamics Simulation')
parser.add_argument('--gen',default='md.traj', help='atomic configuration')
parser.add_argument('--index',default=-1,type=int, help='the index in atomic configurations')
parser.add_argument('--step',default=100,type=int, help='the step of MD simulations')
parser.add_argument('--T',default=300,type=int ,help='the Temperature of MD simulations')
args = parser.parse_args(sys.argv[1:])


def moleculardynamics():
    # opt(gen=gen)
    atoms = read(args.gen,index=args.index) # *[2,2,1]
    # ao    = AtomDance(atoms,bondTole=1.35)
    # atoms = ao.bond_momenta_bigest(atoms)
    
    irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=args.step,gen=args.gen,Tmax=10000,
                 # freeatoms=[4,5],beta=0.77,
                 ro=0.8,rmin=0.5,initT=args.T,
                 ffield='ffield.json',
                 nn=False)
    irmd.run()
    mdsteps= irmd.step
    Emd  = irmd.Epot
    irmd.close()
    

if __name__ == '__main__':
   ''' run this script as :
        ./md.py --s=100 --g=md.traj 
        s 模拟步长
        g 初始结构'''
   moleculardynamics()

