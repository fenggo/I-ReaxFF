#!/usr/bin/env python
import random
from os import system
import subprocess
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
from ase.io import read # ,write
from ase.data import atomic_numbers, atomic_masses
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammpsrun import LAMMPS
# from irff.irff_np import IRFF_NP
from irff.md.gulp import write_gulp_in,arctotraj,get_reax_energy,get_md_results,plot_md,xyztotraj
from irff.md.lammps import get_lammps_forces
from irff.md.lammps import writeLammpsData,writeLammpsIn,get_lammps_thermal,lammpstraj_to_ase
# from deepmd.calculator import DP

# def get_deepmd_energy(atoms):
#     calc = DP(model="EMFF-2025_V1.0.1.pb")
#     atoms.calc = calc
#     calc.set(tmp_dir='./')
#     calc.set(keep_tmp_files=False)
#     calc.set(keep_alive=False)
#     return atoms.get_potential_energy()

def get_gulp_energy(atoms,n=4,libfile='reaxff_nn.lib'):
    ''' get the energy of the atoms using gulp, after structure optimization '''
   #  write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',
   #              lib=libfile)
   #  system('gulp<inp-gulp>out')
   #  e_ = get_reax_energy(fo='out')
    runword= 'opti conp qiterative stre atomic_stress'
    write_gulp_in(atoms,runword=runword,
                  maxcyc=1000,pressure=0.0,
                  lib='reaxff_nn.lib')
    print('\n-  running gulp optimize ...')
    if n==1:
       system('gulp<inp-gulp>gulp.out')
    else:
       system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(n))
    # xyztotraj('his.xyz',mode='w',traj='md.traj',checkMol=c,scale=False) 
    atoms = arctotraj('his_3D.arc',traj='md.traj')
    return atoms

def get_lammps_energy_mtp(atoms,n=4):
    pair_style = 'mlip load_from=pot.almtp'
    pair_coeff = '* * # C O N H'
    units      = "metal"
    atom_style = 'atomic'

    writeLammpsData(atoms,data='data.lammps',specorder=['C','O','N','H'],
                    masses={'Al':26.9820,'C':12.0000,'H':1.0080,'O':15.9990,
                             'N':14.0000,'F':18.9980},
                    force_skew=False,
                    velocities=False,units=units,atom_style=atom_style)
    writeLammpsIn(log='lmp.log',timestep=0.1,total=2000,restart=None,
              dump_interval=10,
              species=['C','O','N','H'],
              pair_style = pair_style,  # without lg set lgvdw no
              pair_coeff = pair_coeff,
              fix = ' ',
              fix_modify = ' ',
              minimize   = '1e-5 1e-5 2000 2000',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',units=units,atom_style=atom_style,
              restartfile='restart')
    print('\n-  running lammps minimize ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',units=units)
   #  line = subprocess.check_output('grep \"Total Energy:\" lmp.log',shell=True)
   #  e_ = float(line.split()[-1])
   #  atoms.energy = e_
    return atoms

def trajplot(traj='siesta.traj',n=8,i=0,j=1,n_struct=100):
    images_        = Trajectory(traj)
    step,e1,e2,e= [],[],[],[]
    e3,e4          = [],[]
    r              = []
    # 随机抽样100个结构
    n_images = len(images_)
    masses = np.sum(images_[0].get_masses())
    if n_images > n_struct:
        sampled_indices = random.sample(range(n_images), n_struct)
        images = [images_[idx] for idx in sampled_indices]
    else:
        images = images_
    fe = open("energy.txt","w")
    for i_,atoms in enumerate(images):
        step.append(i_)
        e.append(atoms.get_potential_energy())
        atoms_rnn = get_gulp_energy(atoms,n=8,libfile='reaxff_nn.lib')
        #### calculate density 
    
        volume = atoms_rnn.get_volume()
        density_rnn = masses/volume/0.602214129

        e1.append(atoms_rnn.get_potential_energy())
        # e2.append(get_lammps_forces(atoms).get_potential_energy())
        atoms_mtp = get_lammps_energy_mtp(atoms,n=8)

        volume = atoms_mtp.get_volume()
        density_mtp = masses/volume/0.602214129
        
        e3.append(atoms_mtp.get_potential_energy())
        # e4.append(get_deepmd_energy(atoms))
        print(e[-1], e1[-1],e3[-1],density_rnn,density_mtp,file=fe)
        
    fe.close()
    e1  = np.array(e1) - min(e1)
    # e2  = np.array(e2) - min(e2)
    e3  = np.array(e3) - min(e3)
    # e4  = np.array(e4) - min(e4)
    e   = np.array(e)  - min(e)
    sort_idx = np.argsort(e)
    # e   = e[sort_idx]
    # e1  = e1[sort_idx]
    # e2  = e2[sort_idx]
    # e3  = e3[sort_idx]

    if i==0 and j==0:
       r = [i_ for i_ in range(len(e))]
    plt.figure()   
    plt.ylabel(r'$Relative\ Energy\ (eV)$')
    plt.xlabel(r'$Time\ Step\ (fs)$')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_position(('data',0))
    # ax.spines['bottom'].set_position(('data', 0))
    plt.plot(r,e1,alpha=0.8,
             linestyle='-',marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='b',markersize=10,
             color='b',label=r'$ReaxFF-nn$')

    plt.plot(r,e3,alpha=0.8,
             linestyle='-',marker='^',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='g',markersize=10,
             color='g',label=r'$MTP$')
    
    # plt.plot(r,e4,alpha=0.8,
    #          linestyle='-',marker='<',markerfacecolor='none',
    #          markeredgewidth=1,markeredgecolor='k',markersize=10,
    #          color='k',label=r'$DeepMD$')
    
    plt.plot(r,e,alpha=0.8,
             linestyle='-',marker='s',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='r',markersize=10,
             color='r',label=r'$DFT(SIESTA)$')
    # pdiff = np.abs(pdft - preax)
    # plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='palegreen',
    #                  alpha=0.2)
    # plt.text( 0.0, 0.5, '%.3f' %e_max, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    plt.savefig('{:s}'.format(traj.replace('traj','svg')),transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''

   parser = argparse.ArgumentParser(description='stretch molecules')
   parser.add_argument('--t', default='structures.traj',type=str, help='trajectory file')
   parser.add_argument('--i', default=0,type=int, help='atom i')
   parser.add_argument('--j', default=0,type=int, help='atom j')
   parser.add_argument('--n', default=8,type=int, help='number of cpu to be used')
   parser.add_argument('--ns', default=100,type=int, help='number of structures to be plotted')
   args = parser.parse_args(sys.argv[1:])
   trajplot(traj=args.t,i=args.i,j=args.j,n=args.n,n_struct=args.ns)

