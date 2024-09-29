#!/usr/bin/env python
import argh
import argparse
from os import system,popen
import time
import numpy as np
from ase.io import read # ,write
from ase import Atoms
from irff.md.gulp import write_gulp_in,arctotraj,get_md_results,plot_md,xyztotraj


def nvt(T=350.0,time_step=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
        x=1,y=1,z=1,n=1,lib='reaxff_nn'):
    atoms = read(gen,index=i)*(x,y,z)
    write_gulp_in(atoms,runword='md qiterative conv',
                  T=T,
                  ensemble='nvt',
                  tau_thermostat=0.1,
                  time_step=time_step,
                  tot_step=step,
                  lib=lib)
    print('\n-  running gulp nvt ...')
    if n==1:
       system('gulp<inp-gulp>gulp.out')
    else:
       system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(n))
    xyztotraj('his.xyz',mode=mode,traj='md.traj', checkMol=c,scale=False)

def phonon(T=300,gen='siesta.traj',step=200,i=-1,l=0,c=0,p=0.0,
        x=1,y=1,z=1,n=1,t=0.00001,lib='reaxff_nn'):
    A = read(gen,index=i)
    # A = press_mol(A)
    runword= 'opti conv qiterative prop phonons thermal num3'
    write_gulp_in(A,inp='inp-phonon',
                  runword=runword,
                  T=T,maxcyc=step,pressure=p,
                  gopt=t,
                  supercell='zyx {:d} {:d} {:d}'.format(x,y,z),
                  lib=lib)
    print('\n-  running gulp phonon calculation ...')
    if n==1:
       system('gulp<inp-phonon>phonon.out')
    else:
       system('mpirun -n {:d} gulp<inp-phonon>phonon.out'.format(n))
    atoms = arctotraj('his_3D.arc',traj='md.traj',checkMol=c)

def opt(T=300,gen='siesta.traj',step=200,i=-1,l=0,c=0,p=0.0,
        x=1,y=1,z=1,n=1,t=0.00001,output=None,lib='reaxff_nn'):
    A = read(gen,index=i)*(x,y,z)
    # A = press_mol(A) 

    if l==1 or p>0.0000001:
       runword= 'opti conp qiterative stre atomic_stress'
    elif l==0:
       runword='opti conv qiterative'
    if output=='shengbte':
       runword= 'opti conv qiterative prop phonons thermal num3'
    write_gulp_in(A,runword=runword,
                  T=T,maxcyc=step,pressure=p,
                  output=output,
                  gopt=t,
                  lib=lib)
    print('\n-  running gulp optimize ...')
    if n==1:
       system('gulp<inp-gulp>gulp.out')
    else:
       system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(n))
    # xyztotraj('his.xyz',mode='w',traj='md.traj',checkMol=c,scale=False) 
    atoms = arctotraj('his_3D.arc',traj='md.traj',checkMol=c)
    if x>1 or y>1 or z>1:
       ncell     = x*y*z
       natoms    = int(len(atoms)/ncell)
       species   = atoms.get_chemical_symbols()
       positions = atoms.get_positions()
       # forces  = atoms.get_forces()
       cell      = atoms.get_cell()
       cell      = [cell[0]/x, cell[1]/y,cell[2]/z]
       u         = np.linalg.inv(cell)
       pos_      = np.dot(positions[0:natoms], u)
       posf      = np.mod(pos_, 1.0)          # aplling simple pbc conditions
       pos       = np.dot(posf, cell)
       atoms     = Atoms(species[0:natoms],pos,#forces=forces[0:natoms],
                         cell=cell,pbc=[True,True,True])
   
    atoms.write('POSCAR.unitcell')
    # return atoms

def sheng(T=300,gen='siesta.traj',step=200,i=-1,l=0,c=0,p=0.0,
        x=1,y=1,z=1,n=1,t=0.00001,output='shengbte',lib='reaxff_nn'):
    A = read(gen,index=i)
    # A = press_mol(A)

    if l==1 or p>0.0000001:
       runword= 'opti conp qiterative stre atomic_stress'
    elif l==0:
       runword='opti conv qiterative'
    if output=='shengbte':
       runword= 'opti conv qiterative prop phonons thermal num3'
    write_gulp_in(A,inp='inp-sheng',
                  runword=runword,
                  T=T,maxcyc=step,pressure=p,
                  output=output,
                  gopt=t,
                  supercell='zyx {:d} {:d} {:d}'.format(x,y,z),
                  lib=lib)
    print('\n-  running gulp optimize ...')
    if n==1:
       system('gulp<inp-sheng>sheng.out')
    else:
       system('mpirun -n {:d} gulp<inp-sheng>sheng.out'.format(n))
    # xyztotraj('his.xyz',mode='w',traj='md.traj',checkMol=c,scale=False)
    atoms = arctotraj('his_3D.arc',traj='md.traj',checkMol=c)

def traj(inp='inp-gulp',c=0):
    #xyztotraj('his.xyz',inp=inp,mode='w',traj='md.traj',scale=False)
    arctotraj('his_3D.arc',traj='md.traj',checkMol=c)

def plot(out='out'):
    E,Epot,T,P = get_md_results(out=out)
    plot_md(E,Epot,T,P,show=True)

def w(T=350,gen='siesta.traj',step=200,i=-1,l=0,c=0,
        x=1,y=1,z=1,lib='reaxff_nn'):
    A = read(gen,index=i)*(x,y,z)
    # A = press_mol(A)
    if l==0:
       runword='opti conv qiterative' # conjugate dynamical_matrix
    elif l==1:
       runword= 'opti conp qiterative stre atomic_stress'

    write_gulp_in(A,runword=runword,
                  T=T,maxcyc=step,
                  lib=lib)
    print('\n-  write gulp input ...')

if __name__ == '__main__':
   ''' use commond like: 
          ./gmd.py nvt --T=2800 --s=5000 --g=*.gen 
       to run this script.
       ---------------------------------------------
       nvt: NVT MD simulation
       opt: structure optimization
       w  : write the gulp input file
       --g: the atomic structure file 

   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [opt,nvt,plot,traj,w,sheng,phonon])
   argh.dispatch(parser)

