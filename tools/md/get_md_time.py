#!/usr/bin/env python
import time
from os import system
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.data import atomic_numbers, atomic_masses
from irff.md.gulp import write_gulp_in
from irff.md.lammps import writeLammpsData,writeLammpsIn


def gulp_nvt(atoms,T=350,time_step=0.1,step=100,n=1,lib='reaxff_nn'):
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


def lammps_nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,model='reaxff-nn',c=0,
        free=' ',dump_interval=10,
        n=1,lib='ffield',thermo_fix=None):
    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    sp      = ' '.join(species)
    freeatoms = free.split()
    freeatoms = [int(i)+1 for i in freeatoms]
    masses    = {s:atomic_masses[atomic_numbers[s]] for s in species }
    
    if model == 'reaxff':
       pair_style = 'reaxff control checkqeq yes'   # without lg set lgvdw no
       pair_coeff = '* * {:s} {:s}'.format(lib,sp)
       units      = "real"
       atom_style = 'charge'
    elif model == 'quip':
       pair_style = 'quip'  
       # lib        = 'Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\"'
       pair_coeff = '* * Carbon_GAP_20_potential/Carbon_GAP_20.xml \"\" 6'
       units      = "metal"
       atom_style = 'atomic'
    elif model == 'dp':
       pair_style = 'deepmd csp2-compress.pb out_freq 10'  
       pair_coeff = '* * '
       units      = "metal"
       atom_style = 'atomic'
    elif model == 'ace':
       pair_style = 'hybrid/overlay pace table linear 10000'  
       pair_coeff = ['* * pace c_ace.yace C','* * table d2_short.table D2 9.0']
       units      = "metal"
       atom_style = 'atomic'
    else:
       pair_style = 'reaxff control nn yes checkqeq yes'   # without lg set lgvdw no
       pair_coeff = '* * {:s} {:s}'.format(lib,sp)
       units      = "real"
       atom_style = 'charge'
        
    if thermo_fix is None:
       thermo_fix = 'fix   1 all nvt temp {:f} {:f} {:d} '.format(T,T,tdump) 

    thermo_style = 'thermo_style  custom step temp epair etotal press vol \
       cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz'
     
    data = 'data.lammps'
    writeLammpsData(atoms,data='data.lammps',specorder=None, 
                    masses=masses,
                    force_skew=False,
                    velocities=False,units=units,atom_style=atom_style)
   

    writeLammpsIn(log='lmp.log',timestep=timestep,total=step,restart=None,
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
    # atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',recover=c,units=units)

'''
      a single MD step run time test 
'''

fd = open('md_time.txt','w')

time_lammps    = 0.0
time_lammps_nn = 0.0
time_lammps_ace = 0.0
time_lammps_quip = 0.0
time_lammps_dp = 0.0

for i in [2,4,6,8]: 
    atoms    = read('gp8.gen')*(i,i,1)
    natom    = len(atoms)

    # start_time = time.time()
    # gulp_nvt(atoms=atoms,T=300,time_step=0.1,step=100,
    #          n=8,lib='reaxff_nn')
    # time_gulp_nn = (time.time() - start_time)/100.0

    # start_time = time.time()
    # gulp_nvt(atoms=atoms,T=300,time_step=0.1,step=100,
    #          n=8,lib='reaxff')
    # time_gulp = (time.time() - start_time)/100.0
    
    start_time = time.time()
    lammps_nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,model='dp',c=0,
               free=' ',dump_interval=10,
               n=8,thermo_fix=None)
    time_lammps_dp = (time.time() - start_time)/100.0
    
    start_time = time.time()
    lammps_nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,model='ace',c=0,
               free=' ',dump_interval=10,
               n=8,thermo_fix=None)
    time_lammps_ace = (time.time() - start_time)/100.0
    
    start_time = time.time()
    lammps_nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,model='quip',c=0,
               free=' ',dump_interval=10,
               n=8,thermo_fix=None)
    time_lammps_quip = (time.time() - start_time)/100.0
    
    start_time = time.time()
    lammps_nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,model='reaxff-nn',c=0,
               free=' ',dump_interval=10,
               n=8,lib='ffield',thermo_fix=None)
    time_lammps_nn = (time.time() - start_time)/100.0
    
    start_time = time.time()
    lammps_nvt(atoms,T=350,tdump=100,timestep=0.1,step=100,model='reaxff',c=0,
               free=' ',dump_interval=10,
               n=8,lib='ffield.reax.rdx',thermo_fix=None)
    time_lammps = (time.time() - start_time)/100.0
    
    print(natom,time_lammps,time_lammps_nn,time_lammps_quip,time_lammps_ace,time_lammps_dp,file=fd)

fd.close()

