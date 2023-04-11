#!/usr/bin/env python
from irff.molecule import packmol
from ase.io import read,write
from ase import build


A = read('POSCAR')
# build.make_supercell(A,[2,2,2])
write('poscar.gen',A*(2,2,2))

packmol(strucs=['poscar.gen'],
        supercell=[3,3,2],
        w=True)
