#!/usr/bin/env python
from os import getcwd,chdir,mkdir,system
from os.path import exists
import argh
import argparse
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.data import chemical_symbols
from irff.dft.siesta import siesta_md,siesta_opt,write_siesta_in


def opt_structure(ncpu=8,T=2500,us='F',gen='poscar.gen',l=0,i=-1,step=200):
    if exists('siesta.MDE') or exists('siesta.MD_CAR'):
       system('rm siesta.MDE siesta.MD_CAR')
    A = read(gen,index=i)
    # A = press_mol(A)
    print('\n-  running siesta opt ...')
    vc = 'true' if l else 'false'
    siesta_opt(A,ncpu=ncpu,us=us,VariableCell=vc,tstep=step,
               xcf='GGA',xca='PBE',basistype='split')

def calc_strutures(structures,step=50,ncpu=8):
    for atoms in structures:
        siesta_opt(atoms,ncpu=ncpu,us='F',VariableCell='true',tstep=step,
                   xcf='GGA',xca='PBE',basistype='split')


if __name__=='__main__': 
   images = Trajectory('../../structures.traj')
   structures = []
   strucs     = []
   cwd        = getcwd()

   for s in structures:
       if exists(s):
          continue
       else:
          mkdir(s)

       strucs.append(images[s])

   calc_strutures(strucs,step=50,ncpu=8)
   
