#!/usr/bin/env python
import argh
import argparse
from os import system #,popen
from ase.io import read # ,write
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase


def nvt(T=350,tdump=100,timestep=0.1,step=100,gen='poscar.gen',i=-1,model='reaxff-nn',c=0,
        free=' ',dump_interval=10,
        x=1,y=1,z=1,n=1,lib='ffield',thermo_fix=None):
    atoms = read(gen,index=i)*(x,y,z)
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    freeatoms = free.split()
    freeatoms = [int(i)+1 for i in freeatoms]
    
    if model == 'quip':
       pair_style = 'quip'  
       lib        = 'Carbon_GAP_20.xml \"\"'
       pair_coeff = '* * {:s} {:s}'.format(lib,sp)
       units      = "metal"
       atom_style = 'atomic'
    else:
       pair_style = 'reaxff control nn yes checkqeq yes'   # without lg set lgvdw no
       pair_coeff = '* * {:s} {:s}'.format(lib,sp)
       units      = "real"
       atom_style = 'charge'
    if thermo_fix is None:
       thermo_fix = 'fix   1 all nvt temp {:f} {:f} {:f} '.format(T,T,tdump) 

    thermo_style = 'thermo_style  custom step temp epair etotal press vol \
       cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz'
    writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units=units,atom_style=atom_style)
    
    writeLammpsIn(log='lmp.log',timestep=timestep,total=step,restart=None,
              species=species,
              pair_coeff = pair_coeff,
              pair_style = pair_style,  # without lg set lgvdw no
              fix = thermo_fix,
              freeatoms=freeatoms,natoms=len(atoms),
              fix_modify = ' ',
              dump_interval=dump_interval,more_commond = ' ',
              thermo_style =thermo_style,
              data='data.lammps',
              restartfile='restart')
    print('\n-  running lammps nvt ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    lammpstraj_to_ase('lammps.trj',inp='in.lammps',recover=c)

def npt(T=350,tdump=100,timestep=0.1,step=100,gen='poscar.gen',i=-1,model='reaxff-nn',c=0,
        p=0.0,x=1,y=1,z=1,n=1,lib='ffield',free=' ',dump_interval=10):
    thermo_fix = 'fix   1 all npt temp {:f} {:f} {:d} iso {:f} {:f} {:d}'.format(T,
                  T,tdump,p,p,tdump)
    nvt(T=T,tdump=tdump,timestep=timestep,step=step,gen=gen,i=i,model=model,c=c,
        free=free,dump_interval=dump_interval,
        x=x,y=y,z=z,n=n,lib=lib,thermo_fix=thermo_fix)

def opt(T=350,timestep=0.1,step=100,gen='poscar.gen',i=-1,model='w',c=0,
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
    lammpstraj_to_ase('lammps.trj',inp='in.lammps',recover=c)

def traj(inp='in.lammps',s=0,e=0,c=0):
    if e==0:
       atomid = None
    else:
       atomid = (s,e)
    lammpstraj_to_ase('lammps.trj',inp=inp,atomid=atomid,recover=c)

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
   argh.add_commands(parser, [opt,npt,nvt,plot,traj,w])
   argh.dispatch(parser)

