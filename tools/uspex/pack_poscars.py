#!/usr/bin/env python
from os import getcwd,listdir
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
# from irff.molecule import press_mol

cdir    = getcwd()
files   = listdir(cdir)
 
poscars = []

for fil in files:
    f = fil.split('.')
    if len(f)>=1:
       if f[0]=='POSCAR':
          poscars.append(fil)

fposcars = open('POSCARS','a')

for p in poscars:
    p_ = p.split('.')
    print(p_)
    if len(p_)>1:
       i_ = p_[1]
    else:
       continue
    atoms = read(p)
    structure = AseAtomsAdaptor.get_structure(atoms)
    structure.to(filename="POSCAR")
    cell = atoms.get_cell()
    angles = cell.angles()
    lengths = cell.lengths()
    with open('POSCAR','r') as f:
         lines = f.readlines()
     
    card = False
    for i,line in enumerate(lines):
        if line.find('direct')>=0:
           card = True
        if card and line.find('direct')<0:
           print(line[:-3],file=fposcars)
        elif i==0:
           print('EA{:s} {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f} Sym.group: 1'.format(i_,
                   lengths[0],lengths[1],lengths[2],
                   angles[0],angles[1],angles[2]),file=fposcars)
        else:
           print(line[:-1],file=fposcars)
fposcars.close()
    