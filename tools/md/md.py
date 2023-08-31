#!/usr/bin/env python3
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


help_ = './md.py --s=100 --g=md.traj --f=[1,2]'

parser = argparse.ArgumentParser(description=help_)
parser.add_argument('--gen',default='md.traj', help='atomic configuration')
parser.add_argument('--index',default=-1,type=int, help='the index in atomic configurations')
parser.add_argument('--step',default=100,type=int, help='the step of MD simulations')
parser.add_argument('--T',default=300,type=int ,help='the Temperature of MD simulations')
parser.add_argument('--r',default=0,type=int ,help='if run relax mode, i.e. relax the structure')
parser.add_argument('--b',default=0.96,type=float,help='the relax parameter')
parser.add_argument('--f',default=[],type=list,help='free atoms')
parser.add_argument('--m',default=1,type=int,help='manybody interaction flag')
args = parser.parse_args(sys.argv[1:])


def moleculardynamics():
    # opt(gen=gen)
    atoms = read(args.gen,index=args.index)#*[2,2,1]
    # ao    = AtomDance(atoms,bondTole=1.35)
    # atoms = ao.bond_momenta_bigest(atoms)
    
    f_= args.f if args.f else None
    nomb = False if args.m else True 
    
    irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=args.step,gen=args.gen,Tmax=10000,
                 freeatoms=f_,beta=args.b,
                 ro=0.8,rmin=0.5,initT=args.T,
                 ffield='ffield.json',
                 nomb=nomb,nn=True)
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

