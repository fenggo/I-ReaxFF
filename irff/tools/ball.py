#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from os.path import exists
from os import system,getcwd,chdir
from ase.io import read,write
from irff.emdk import get_structure


get_structure(struc='HMX',supercell=[10,10,10])

A = read('card.gen')
cell = A.get_cell()
atom_name = A.get_chemical_symbols()
natom     = len(atom_name)

print(cell)
print('\nnumber of atoms:',natom)



