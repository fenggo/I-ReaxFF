#!/usr/bin/env python
from os.path import isfile
from os import system, getcwd,listdir
import sys
import argparse
import numpy as np
from ase.io import read

''' A work flow in combination with USPEX '''

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--n',default=1,type=int, help='the number of cpu used in this calculation')
parser.add_argument('--d',default=1.85,type=float, help='the minimal density')
args = parser.parse_args(sys.argv[1:])
 

def run_gulp(n=1,inp='input'):
    if n==1:
       system('gulp<{:s}>output'.format(inp)) 
    else:
       system('mpirun -n {:d} gulp<{:s}>output'.format(n,inp))  # get initial crystal structure

def write_input(inp='inp-grad'):
    with open('input','r') as f:
      lines = f.readlines()
    with open(inp,'w') as f:
      for i,line in enumerate(lines):
          if i==0 :
             print('grad nosymmetry conv qiterative',file=f)
          # elif line.find('maxcyc')>=0:
          #    print('maxcyc 0',file=f)
          else:
             print(line.rstrip(),file=f)
def write_output():
    with open('output','r') as f:
         for line in f.readlines():
             if line.find('Total lattice energy')>=0 and line.find('eV')>0:
                e = float(line.split()[4])
    with open('output','w') as f:
         print('  Cycle:      0 Energy:       {:f}'.format(e),file=f)



write_input(inp='inp-grad')
run_gulp(n=args.n,inp='inp-grad')
write_output()

atoms = read('gulp.cif')
masses = np.sum(atoms.get_masses())
volume = atoms.get_volume()
density = masses/volume/0.602214129

# if density <= args.d:
#    run_gulp(n=args.n,inp='input')
