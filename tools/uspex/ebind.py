#!/usr/bin/env python
# coding: utf-8
import sys
import argparse
import numpy as np
import copy
import json as js
from os import system
from ase.io import read
from ase.io.trajectory import TrajectoryWriter #,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.molecule import Molecules,enlarge # SuperCell,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol
from irff.md.gulp import write_gulp_in,get_reax_energy,opt
''' scale the crystal box, while keep the molecule structure unchanged
'''

parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--i', default=0,type=int, help='index of atomic frame')
args = parser.parse_args(sys.argv[1:])

lf = open('ffield.json','r')
j = js.load(lf)
lf.close()

# A = read(args.g,index=args.i)
# A = press_mol(A)
A = opt(gen=args.g,i=args.i,step=1000,l=1,lib='reaxff_nn',n=1)
# A = read('POSCAR.unitcell')

x = A.get_positions()
m = np.min(x,axis=0)
x_ = x - m
A.set_positions(x_)

# print(j['rcutBond'])

m_  = Molecules(A,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,"others": 1.68},check=True)
nmol = len(m_)



ir = IRFF_NP(atoms=A,
             libfile='ffield.json',
             nn=True)

print('\nnumber of molecules:',nmol)

ff = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9 ,2.0,2.5,3.0,3.5,4.0] #,1.9 ,2.0,2.5,3.0,3.5,4.0
# ff = [3]
cell = A.get_cell()
e  = []
eg = []

with TrajectoryWriter('md.traj',mode='w') as his:
    for f in ff:
        m = copy.deepcopy(m_)
        _,A = enlarge(m,cell=cell,fac=f,supercell=[1,1,1])
        ir.calculate(A)
        e.append(ir.E)
		
        write_gulp_in(A,runword='gradient nosymmetry conv qite verb',
                      lib='reaxff_nn')
        system('gulp<inp-gulp>out')
        (e_,eb_,el_,eo_,eu_,ea_,ep_,
         etc_,et_,ef_,ev_,ehb_,ecl_,esl_)= get_reax_energy(fo='out')
        eg.append(e_)
		 
        print('The binding energy: ',e[0]-e[-1],'average:', (e[0]-e[-1])/nmol,'gulp:', (eg[0]-eg[-1])/nmol)
        A.calc = SinglePointCalculator(A,energy=ir.E)
        his.write(atoms=A)
 
