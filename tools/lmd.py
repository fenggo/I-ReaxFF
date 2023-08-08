#!/usr/bin/env python
import argh
import argparse
from os import system #,popen
from ase.io import read # ,write
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,LammpsHistory


def nvt(T=350,timestep=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
        x=1,y=1,z=1,n=1,lib='ffield'):
    atoms = read(gen,index=i)*(x,y,z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units="real",atom_style='charge')
    writeLammpsIn(log='lmp.log',timestep=timestep,total=step,restart=None,
              species=species,
              pair_coeff ='* * {:s} {:s}'.format(lib,sp),
              pair_style = 'reaxff control nn yes checkqeq yes',  # without lg set lgvdw no
              fix = 'fix   1 all nvt temp 300 300 100.0 ',
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              restartfile='restart.eq')
    print('\n-  running lammps nvt ...')
    if n==1:
       system('/home/feng/mlff/lammps/src/lmp_serial<in.lammps>out')
       # system('/home/feng/lammps/src/lmp_serial<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps<in.lammps>out'.format(n))
    LammpsHistory('lammps.trj',inp='in.lammps')


def opt(T=350,gen='siesta.traj',step=200,i=-1,l=0,c=0,
        x=1,y=1,z=1,n=1,lib='reaxff_nn'):
    A = read(gen,index=i)*(x,y,z)
    # A = press_mol(A)
    writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units="real",atom_style='charge')
    if l==0:
       runword='opti conv qiterative'
    elif l==1:
       runword= 'opti conp qiterative stre atomic_stress'

    writeLammpsIn(log='lmp.log',timestep=0.1,total=200,restart=None,
              pair_coeff ='* * ffield C H O N',
              pair_style = 'reaxff control checkqeq yes',  # without lg set lgvdw no
              fix = 'fix   1 all nvt temp 300 300 100.0 ',
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              restartfile='restart.eq')
    print('\n-  running gulp optimize ...')
    if n==1:
       system('/home/feng/mlff/lammps/src/lmp_serial<inp.lammps>out')
    else:
       system('mpirun -n {:d} /home/feng/mlff/lammps/src/lmp_serial<inp.lammps>out'.format(n))
    # xyztotraj('his.xyz',mode=mode,traj='md.traj', checkMol=c,scale=False)

def traj(inp='inp-gulp'):
    LammpsHistory('lammps.trj',inp='in.lammps')

def plot(out='out'):
    get_lammps_thermal(logname='lmp.log',supercell=[1,1,1])

def w(T=350,gen='siesta.traj',step=200,i=-1,l=0,c=0,
        x=1,y=1,z=1,lib='reaxff_nn'):
    A = read(gen,index=i)*(x,y,z)
    # A = press_mol(A)
    if l==0:
       runword='opti conv qiterative' # conjugate dynamical_matrix
    elif l==1:
       runword= 'opti conp qiterative stre atomic_stress'

    writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units="real",atom_style='charge')
    writeLammpsIn(log='lmp.log',timestep=0.1,total=200,restart=None,
              pair_coeff ='* * ffield C H O N',
              pair_style = 'reaxff control checkqeq yes',  # without lg set lgvdw no
              fix = 'fix   1 all nvt temp 300 300 100.0 ',
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              restartfile='restart.eq')
    print('\n-  write gulp input ...')

if __name__ == '__main__':
   ''' use commond like: 
          ./lmd.py nvt --T=2800 --s=5000 --g=*.gen 
       to run this script.
       ---------------------------------------------
       nvt: NVT MD simulation
       opt: structure optimization
       w  : write the gulp input file
       --g: the atomic structure file 
       --s: MD simulation steps
       --T: MD simulation temperature
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [opt,nvt,plot,traj,w])
   argh.dispatch(parser)


