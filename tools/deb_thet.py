#!/usr/bin/env python
# coding: utf-8
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units,Atoms
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from irff.irff_np import IRFF_NP
from irff.plot.LearningResults import learning_result
# from irff.tools import deb_energy
import matplotlib.pyplot as plt
from irff.plot.deb_bde import deb_energy,deb_bo
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import argparse


def get_theta(atoms):
    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 nn=True)
    ir.calculate(atoms)
     
    for a,angle in enumerate(ir.angs):  
        i,j,k = angle
        print('{:3d} {:3d} {:3d} {:3d}  {:6.4f}  {:6.4f} Dang: {:6.4f} rnlp: {:6.4f} '
              'SBO: {:6.4f} sbo: {:6.4f} pbo: {:6.4f} SBO3: {:6.4f}'.format(a,i,j,k,
               ir.thet0[a],ir.theta[a],ir.dang[a],ir.rnlp[a],ir.SBO[a],ir.sbo[a],ir.pbo[a],
               ir.SBO3[a])) # self.thet0-self.theta
        SBO = ir.sbo[a] - (1.0-ir.pbo[a])*(ir.dang[a]+ir.P['val8']*ir.rnlp[a])
    # print(ir.sbo[a],1.0-ir.pbo[a],ir.dang[a]+ir.P['val8']*ir.rnlp[a])
    # print(SBO)
    # pbo1 = np.exp(-np.power(ir.bo0[0][1],8))
    # pbo2 = np.exp(-np.power(ir.bo0[0][2],8))
    # print(pbo1,pbo2)


parser = argparse.ArgumentParser(description='get theta0')
parser.add_argument('--g',default='md.traj',type=str,help='the atomic gementry file name')
parser.add_argument('--i',default=0,type=int,help='index of trajectory frame')

args = parser.parse_args(sys.argv[1:])

atoms  = read(args.g,args.i)
get_theta(atoms)

# 120*3.141593/180


''' Usage: ./deb_thet.py --g=md.traj --i=-1  
'''
