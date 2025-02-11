#!/usr/bin/env python
# coding: utf-8
import sys
from os import getcwd,chdir,mkdir,system,listdir
from os.path import exists
import argparse
import numpy as np
import copy
import json as js
from os import system
from ase.io import read
from ase.io.trajectory import TrajectoryWriter #,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from irff.molecule import Molecules,moltoatoms
#from irff.md.lammps import writeLammpsData
from irff.irff_np import IRFF_NP
from irff.molecule import press_mol
from irff.md.gulp import write_gulp_in,get_reax_energy,opt
from irff.dft.siesta import siesta_opt,single_point

''' scale the crystal box, while keep the molecule structure unchanged
'''

def cleanup(stru,atoms,sf='opt'):
    system('mv siesta.out siesta-{:s}.out'.format(stru))
    system('mv siesta.MDE siesta-{:s}.MDE'.format(stru))
    system('mv siesta.MD_CAR siesta-{:s}.MD_CAR'.format(stru))
    system('mv siesta.traj id_{:s}.traj'.format(stru))

    system('rm siesta.* ')
    atoms = img[-1]
    atoms.write('POSCAR.{:s}_{:s}'.format(stru,sf))

    masses = np.sum(atoms.get_masses())
    volume = atoms.get_volume()
    density = masses/volume/0.602214129
    return density

parser = argparse.ArgumentParser(description='eos by scale crystal box')
#parser.add_argument('--g', default='md.traj',type=str, help='trajectory file')
parser.add_argument('--i', default=0,type=int, help='index of atomic frame')
parser.add_argument('--n', default=24,type=int, help='number of cpu to be used')
parser.add_argument('--x', default='PBE',type=str, help='which funcitonal to be used')
args = parser.parse_args(sys.argv[1:])

if args.x=='PBE':
   xcf='GGA'
   xca='PBE'
else:
   xcf='VDW'
   xca='DRSLL'

cdir   = getcwd()
ids    = listdir(cdir)

if not exists('ebind.dat'):
   with open('ebind.dat','w') as fd:
        print('# Crystal_id lattice_energy molecular_gas_energy binding_energy average_ebind density',file=fd)

moleclue_energy = {'C2N4H4O4'   : -3188.837170,
                   'C6N12O12H6' : -9464.452610}  ## eV

for stru in ids:
    if stru.endswith('.gen') or stru.endswith('.traj') or stru.startswith('POSCAR'):
        A = read(stru)
        A = press_mol(A)
        x = A.get_positions()
        m = np.min(x,axis=0)
        x_ = x - m
        A.set_positions(x_)
        cell = A.get_cell()

        s  = stru.split('.')[0]
        if s == 'POSCAR':
           s = stru.split('.')[-1]

        work_dir = cdir+'/'+str(s)

        if exists(str(s)):
            continue
        else:
            mkdir(str(s))
        chdir(work_dir)
        system('cp ../*.psf ./')

        img = siesta_opt(A,ncpu=args.n,us='F',VariableCell='true',tstep=300,
                         xcf=xcf,xca=xca,basistype='split')
        atoms   = img[-1]
        density = cleanup(s,atoms) 

        m_     = Molecules(atoms,rcut={"H-O":1.22,"H-N":1.22,"H-C":1.22,"O-O":1.4,
                                  "others": 1.68},check=True)
        nmol   = len(m_)
        print('CY: {:5s}  NM: {:4d}  Density {:8.5f}'.format(s,nmol,density))
        # atoms = single_point(A,id=1,xcf=xcf,xca=xca,
        #                          basistype='split',cpu=args.n)
        e = atoms.get_potential_energy()
        emolecules = 0.0
        for m in m_:
            if m.label not in moleclue_energy:
               m.cell = np.array([[15.0,0.0,0.0],[0.0,15.0,0.0],[0.0,0.0,15.0]])
               atoms = moltoatoms([m])
               img = siesta_opt(atoms,ncpu=args.n,us='F',VariableCell='false',tstep=100,
                                xcf=xcf,xca=xca,basistype='split')
               # atoms = single_point(img[-1],id=1,xcf=xcf,xca=xca,
               #                      basistype='split',cpu=args.n)
               atoms  = img[-1]
               density_ = cleanup(m.label,atoms) 

               emol  = atoms.get_potential_energy()
               moleclue_energy[m.label] = emol
               print('        Molecular Energy {:10s}: {:f}'.format(m.label,emol))
            emolecules += moleclue_energy[m.label]

        chdir(cdir)
        with open('ebind.dat','a') as fd:
            print(s,e,emolecules,e-emolecules,(e-emolecules)/nmol,density,file=fd)
        
        