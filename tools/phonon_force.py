#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system
import sys
import argparse
import numpy as np
from ase.io import read # ,write
from ase.visualize import view
from irff.md.gulp import get_gulp_forces
from irff.md.lammps import get_lammps_forces
from irff.dft.siesta import parse_fdf,parse_fdf_species,write_siesta_in
from irff.irff import IRFF


parser = argparse.ArgumentParser(description='stretch molecules')
parser.add_argument('--n',default=1,type=int, help='displacement number')
parser.add_argument('--f',default='in.fdf',type=str, help='fdf file name')
parser.add_argument('--g',default='POSCAR',type=str, help='geomentry file name')
args    = parser.parse_args(sys.argv[1:])

# system('cp geo.genS-00{:d} geo-s{:d}.gen'.format(args.n,args.n))
atoms = read(args.g)
write_siesta_in(atoms, coord='cart',md=False, opt='CG',
                VariableCell='true', xcf='VDW', xca='DRSLL',
                basistype='DZP')

spec  = parse_fdf_species(fdf='in.fdf')
atoms = parse_fdf('supercell-00{:d}'.format(args.n),spec=spec)
#view(atoms)

# get_gulp_forces([atoms])
atoms = get_lammps_forces(atoms)
forces = atoms.get_forces()

with open('Forces.FA', 'w') as ff:
    print(len(atoms), file=ff)
    for i, f in enumerate(forces):
        print('{:4d} {:12.8f} {:12.8f} {:12.8f}'.format(
              i+1, f[0], f[1], f[2]), file=ff)   

system('mv Forces.FA Forces-00{:d}.FA'.format(args.n))

