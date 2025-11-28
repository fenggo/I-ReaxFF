#!/usr/bin/env python
import subprocess
import argparse
import sys
from os import getcwd,chdir,listdir
from os import getcwd,listdir
from os.path import exists
from ase.io import read
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
# from irff.molecule import press_mol

parser = argparse.ArgumentParser(description='merg the molecular dynamics trajectory')
parser.add_argument('--t',default='Individuals.traj',type=str, help='trajectory file name')
parser.add_argument('--start',default=0,type=int, help='the start frame')
parser.add_argument('--end',default=1000,type=int, help='the end frame')
args = parser.parse_args(sys.argv[1:])

start = args.start
end   = args.end
traj  = args.t

images = Trajectory(traj)
if end > len(images):
   end = len(images)
        
fposcars = open('POSCARS','a')
for i_,atoms in enumerate(images):
    # atoms = read(p)
    if i_<start or i_>=end:
       continue
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
           print('EA{:d} {:.6f} {:.6f} {:.6f} {:.3f} {:.3f} {:.3f} Sym.group: 1'.format(i_+1,
                   lengths[0],lengths[1],lengths[2],
                   angles[0],angles[1],angles[2]),file=fposcars)
        else:
           print(line[:-1],file=fposcars)

    # print('{:s}'.format(i_))
fposcars.close()

