#!/usr/bin/env python
from os import getcwd,chdir,listdir,system
from os import getcwd,listdir
from os.path import exists
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
# from irff.molecule import press_mol

density = 1.81
cdir    = getcwd()

cdir_   = '/'.join(cdir.split('/')[:-1])
# print(cdir_)
dirs    = listdir(cdir_)
for dir_ in dirs:
    d = dir_.split('-')
    if d[0]=='results13' and d[1].isalpha(): # d[1].isalpha(): isalnum()
       if exists('{:s}/{:s}/density.log'.format(cdir_,dir_)):
          with open('{:s}/{:s}/density.log'.format(cdir_,dir_),'r') as f:
           for i,line in enumerate(f.readlines()):
               if i==0:
                  continue
               l = line.split()
               den = float(l[1])
               id_ = l[0]
               if den>=density:
                  print(id_,den)
                  if exists('{:s}/{:s}/{:s}/id_{:s}.traj'.format(cdir_,dir_,id_,id_)):
                     atoms = read('{:s}/{:s}/{:s}/id_{:s}.traj'.format(cdir_,dir_,id_,id_))
                  elif exists('{:s}/{:s}/{:s}/individual_{:s}.traj'.format(cdir_,dir_,id_,id_)):
                     atoms = read('{:s}/{:s}/{:s}/individual_{:s}.traj'.format(cdir_,dir_,id_,id_))
                  else:
                     atoms = read('{:s}/{:s}/{:s}/md_{:s}.traj'.format(cdir_,dir_,id_,id_))
                  atoms.write('{:s}/{:s}/{:s}/POSCAR.{:s}'.format(cdir_,dir_,id_,id_))
                  system('cp {:s}/{:s}/{:s}/POSCAR.{:s} POSCAR.{:s}{:s}'.format(cdir_,dir_,id_,id_,d[1],id_))
    
########### pack to poscars ##########
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
    # print(p_)
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

    # print('{:s}'.format(i_))
fposcars.close()

