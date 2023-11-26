#!/usr/bin/env python
import argh
import argparse
from os import system #,popen
from ase.io import read # ,write
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase


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
              restartfile='restart')
    print('\n-  running lammps nvt ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj',inp='in.lammps')

def npt(T=350,timestep=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
        p=0.0,x=1,y=1,z=1,n=1,lib='ffield'):
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
              fix = 'fix   1 all npt temp {:f} {:f} 100 iso {:f} {:f} 100'.format(T,T,p,p), 
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              restartfile='restart')
    print('\n-  running lammps npt ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj',inp='in.lammps')

def opt(T=350,timestep=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
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
              fix = 'minimize	1e-5 1e-5 2000 2000', 
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              restartfile='restart')
    print('\n-  running lammps minimize ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj',inp='in.lammps')

def msst(T=350,timestep=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
        x=1,y=1,z=1,n=1,
        d='z',v=8.0,q=100,
        lib='ffield',r='restart'):
    atoms = read(gen,index=i)*(x,y,z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    if r == 0:
       r = None
    writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units="real",atom_style='charge')
    writeLammpsIn(log='lmp.log',timestep=timestep,total=step,restart=r,
              species=species,
              pair_coeff ='* * {:s} {:s}'.format(lib,sp),
              pair_style = 'reaxff control nn yes checkqeq yes',  # without lg set lgvdw no
              fix = 'fix msst all msst {:s} {:f} {:f} 100 mu 3e2 tscale 0.01 '.format(d,v,q),
              fix_modify = ' ',
              more_commond = ' ',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',
              restartfile='restart')
    print('\n-  running lammps msst ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj',inp='in.lammps')

def traj(inp='in.lammps',s=0,e=0):
    if e==0:
       atomid = None
    else:
       atomid = (s,e)
    lammpstraj_to_ase('lammps.trj',inp=inp,atomid=atomid)

def plot(out='out'):
    get_lammps_thermal(logname='lmp.log',supercell=[1,1,1])

def w(T=350,timestep=0.1,step=100,gen='poscar.gen',i=-1,mode='w',c=0,
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
              restartfile='restart')
    print('\n-  write lammps input ...')

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
   argh.add_commands(parser, [opt,npt,nvt,msst,plot,traj,w])
   argh.dispatch(parser)

