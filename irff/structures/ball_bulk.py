#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from os.path import exists
from os import system,getcwd,chdir
from ase.io import read,write
from ase import Atoms
from irff.structures import structure
from irff.molecule import Molecules,SuperCell,moltoatoms
from irff.emdk import emdk
import numpy as np



def ball_bulk(kernel=20.0,
              shell=14.0,
              shell_thickness=5.0,
              supercell1=[10,10,10],
              supercell2=[28,28,28],
              gap=6.0):
    A = structure('HMX')
    x = A.get_positions()
    m = np.min(x,axis=0)
    x_ = x - m
    A.set_positions(x_)

    M = Molecules(A)
    nmol = len(M)
    print('\nnumber of molecules:',nmol)

    cell = A.get_cell()
    m,A = SuperCell(M,cell=cell,supercell=supercell1)
    natom = len(A)

    xA = A.get_positions()
    centerA = np.sum(xA,axis=0)/natom
    print('-  center of A',centerA)

    mball = []

    # kernel = 25.0   # kernel
    for m_ in m:
        dist = np.sqrt(np.sum(np.square(m_.center-centerA)))
        if dist<kernel:
           mball.append(m_)

    A = moltoatoms(mball)
    atom_name = A.get_chemical_symbols()
    natom     = len(atom_name)
    x         = A.get_positions()
    print('\nnumber of atoms of A:',natom)

    m = np.min(x,axis=0)
    x = x - m
    A.set_positions(x)

    nmol = len(m)
    print('\nnumber of molecules:',nmol)

    B = structure('Al')
    B = B*supercell2

    cell = B.get_cell()

    atom_name2= B.get_chemical_symbols()
    natom     = len(atom_name2)


    print(cell)
    print('\nnumber of atoms of B:',natom)

    xB = B.get_positions()

    centerB = np.sum(xB,axis=0)/natom
    print('-  center of B',centerB)

    x_ = xB - centerB + centerA + [ 2.0*kernel+gap ,0.0,0.0]
    B.set_positions(x_)

    center_ = np.sum(x_,axis=0)/natom
    print('-  new center of B',center_)

    # shell = 28.0
    xB = []
    x  = list(x)

    for i,c in enumerate(x_):
        # dist = np.sqrt(np.sum(np.square(c-centerA)))
        # if dist>=shell and dist<=shell+shell_thickness: 
        # print(c)
        if (c[0]>=2.0*kernel + gap and c[0]<=4.0*kernel + gap and 
            c[1]>=0.0 and c[1]<=2.0*kernel and c[2]>=0.0 and c[2]<=2.0*kernel): 
           atom_name.append(atom_name2[i])
           x.append(c)

    cell=[[4.0*kernel+gap*2,0.0,0.0],
               [0.0,2.0*kernel+gap,0.0],
               [0.0,0.0,2.0*kernel+gap]]
    new = Atoms(atom_name,x,cell=cell,pbc=[True,True,True])
    new.write('ball_bulk.gen')

    emdk(cardtype='xyz',cardfile='ball_bulk.xyz',
         cell=cell,
         element='5 C H O N Al',
         masses='12.000 1.008 15.999 14.0 26.981539',
         supercell=[1,1,1],output='lammpsdata',log='log')


if __name__ == '__main__':
   ball_bulk()


