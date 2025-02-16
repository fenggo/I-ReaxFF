#!/usr/bin/env python
from os.path import isfile
from os import system, getcwd,listdir
import sys
import argparse
import numpy as np
from ase.io import read
from ase import Atoms
from ase.data import atomic_numbers, atomic_masses
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
from irff.md.gulp import write_gulp_in

''' A work flow in combination with USPEX '''

parser = argparse.ArgumentParser(description='./atoms_to_poscar.py --g=siesta.traj')
parser.add_argument('--n',default=1,type=int, help='the number of cpu used in this calculation')
parser.add_argument('--x',default=2,type=int, help='X')
parser.add_argument('--y',default=2,type=int, help='Y')
parser.add_argument('--z',default=2,type=int, help='Z')
parser.add_argument('--p',default=0.0,type=float, help='Pressure')
parser.add_argument('--T',default=300,type=int, help='Temperature')
parser.add_argument('--step',default=10000,type=int, help='Time Step')
parser.add_argument('--d',default=1.75,type=float, help='the minimal density')
args = parser.parse_args(sys.argv[1:])
 
def nvt(T=350,tdump=100,timestep=0.1,step=100,gen='poscar.gen',i=-1,c=0,
        free=' ',dump_interval=10,
        x=1,y=1,z=1,n=1,lib='ffield',thermo_fix=None,
        r=0):
    atoms = read(gen,index=i)*(x,y,z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    freeatoms = free.split()
    freeatoms = [int(i)+1 for i in freeatoms]
    masses    = {s:atomic_masses[atomic_numbers[s]] for s in species }

    pair_style = 'reaxff control nn yes checkqeq yes'   # without lg set lgvdw no
    pair_coeff = '* * {:s} {:s}'.format(lib,sp)
    units      = "real"
    atom_style = 'charge'
 
    if thermo_fix is None:
       thermo_fix = 'fix   1 all nvt temp {:f} {:f} {:d} '.format(T,T,tdump) 

    thermo_style = 'thermo_style  custom step temp epair etotal press vol \
       cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz'

    if r == 0:
       r_=None
       data = 'data.lammps'
       writeLammpsData(atoms,data='data.lammps',specorder=None, 
                       masses=masses,
                       force_skew=False,
                       velocities=False,units=units,atom_style=atom_style)
    else:
       r_ = 'restart'
       data = None

    writeLammpsIn(log='lmp.log',timestep=timestep,total=step,restart=r_,
              species=species,
              pair_style= pair_style,  # without lg set lgvdw no
              pair_coeff=pair_coeff,
              fix = thermo_fix,
              freeatoms=freeatoms,natoms=len(atoms),
              fix_modify = ' ',
              dump_interval=dump_interval,more_commond = ' ',
              thermo_style =thermo_style,
              units=units,atom_style=atom_style,
              data=data,T=T,
              restartfile='restart')
    print('\n-  running lammps ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',recover=c,units=units)
    return atoms

def run_gulp(gen='POSCAR',atoms=None,n=1,inp=None,step=200,l=1,p=0,T=300,t=0.0001,lib='reaxff_nn'):
    if inp is not None:
       if n==1:
          system('gulp<{:s}>output'.format(inp)) 
       else:
          system('mpirun -n {:d} gulp<{:s}>output'.format(n,inp))  # get initial crystal structure
    else:
       if l==1 or p>0.0000001:
          runword= 'opti conp qiterative stre atomic_stress'
       elif l==0:
          runword='opti conv qiterative'
 
       write_gulp_in(atoms,runword=runword,
                  T=T,maxcyc=step,pressure=p,
                  gopt=t,
                  lib=lib)
       print('\n-  running gulp optimize ...')
       if n==1:
          system('gulp<inp-gulp>output')
       else:
          system('mpirun -n {:d} gulp<inp-gulp>output'.format(n))
    # xyztotraj('his.xyz',mode='w',traj='md.traj',checkMol=c,scale=False) 
    # atoms = arctotraj('his_3D.arc',traj='md.traj',checkMol=c)

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

def write_output(e=None):
    if e is None:
       with open('output','r') as f:
         for line in f.readlines():
             if line.find('Total lattice energy')>=0 and line.find('eV')>0:
                e = float(line.split()[4])
    with open('output','w') as f:
         print('  Cycle:      0 Energy:       {:f}'.format(e),file=f)

def npt(T=350,tdump=100,timestep=0.1,step=100,gen='poscar.gen',i=-1,c=0,
        p=0.0,x=1,y=1,z=1,n=1,lib='ffield',free=' ',dump_interval=10,r=0):
    thermo_fix = 'fix   1 all npt temp {:f} {:f} {:d} iso {:f} {:f} {:d}'.format(T,
                  T,tdump,p,p,tdump)
    
    atoms = nvt(T=T,tdump=tdump,timestep=timestep,step=step,gen=gen,i=i,c=c,
                free=free,dump_interval=dump_interval,
                 x=x,y=y,z=z,n=n,lib=lib,thermo_fix=thermo_fix,r=r)
    
    if x>1 or y>1 or z>1:
       ncell     = x*y*z
       natoms    = int(len(atoms)/ncell)
       species   = atoms.get_chemical_symbols()
       positions = atoms.get_positions()
       forces    = atoms.get_forces()
       cell      = atoms.get_cell()
       cell      = [cell[0]/x, cell[1]/y,cell[2]/z]
       u         = np.linalg.inv(cell)
       pos_      = np.dot(positions[0:natoms], u)
       posf      = np.mod(pos_, 1.0)          # aplling simple pbc conditions
       pos       = np.dot(posf, cell)
       atoms     = Atoms(species[0:natoms],pos,#forces=forces[0:natoms],
                         cell=cell,pbc=[True,True,True])
   
    atoms.write('POSCAR.lammps')
    return atoms

write_input(inp='inp-grad')
run_gulp(n=args.n,inp='inp-grad')
write_output()

if density <= args.d:
   atoms = npt(gen='gulp.cif',T=args.T,step=args.step,p=args.p,x=args.x,y=args.y,z=args.z,n=args.n,dump_interval=100)
   run_gulp(n=args.n,atoms=atoms,l=0,step=200)

