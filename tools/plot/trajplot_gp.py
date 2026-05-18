#!/usr/bin/env python
import random
from os import system
import subprocess
from os import getcwd,chdir,mkdir,system
from os.path import exists
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
from ase.io import read # ,write
from ase.data import atomic_numbers, atomic_masses
from sklearn import preprocessing
# from ase.calculators.singlepoint import SinglePointCalculator
# from ase.calculators.lammpsrun import LAMMPS
# from irff.irff_np import IRFF_NP
from irff.md.gulp import write_gulp_in,arctotraj,get_reax_energy,opt
# from irff.md.lammps import get_lammps_forces
from irff.md.lammps import writeLammpsData,writeLammpsIn,lammpstraj_to_ase
# from deepmd.calculator import DP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,DotProduct, WhiteKernel,
                                              ConstantKernel as C,RationalQuadratic,
                                              Matern,
                                              ExpSineSquared)
import pickle

def load_gaussian_process(X,y,y_eng):
    # if not exists('gpr_density.pkl'):
    kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))**2 +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                0.35**2 * Matern(length_scale=[0.0526, 0.0525, 0.0493, 0.01, 0.0439, 0.163, 0.1, 0.1], nu=1.5) +       # 局部耦合
                WhiteKernel(noise_level=0.1)    )                                      # 噪声补偿
    gpr_density = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
    gpr_density.fit(X,y)
    with open('gpr_density.pkl', 'wb') as f:
        pickle.dump(gpr_density, f)
     
    # if not exists('gpr_energy.pkl'):
    kernel = ( 0.00581**2 * DotProduct(sigma_0=0.412, sigma_0_bounds=(1e-4, 50))**2 +   # 线性/多项式趋势 捕捉线性趋势及二阶耦合 (x_i * x_j)
                0.35**2 * Matern(length_scale=[0.0526, 0.0525, 0.0493, 0.01, 0.0439, 0.163, 0.1, 0.1], nu=1.5) +       # 局部耦合
                WhiteKernel(noise_level=0.1)    )                                      # 噪声补偿
    gpr_energy = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)
    gpr_energy.fit(X,y_eng)
    with open('gpr_energy.pkl', 'wb') as f:
        pickle.dump(gpr_energy, f)

    with open('gpcsp.log','w') as fl:
        print(gpr_density.kernel_,file=fl)
        print(gpr_density.log_marginal_likelihood(),file=fl)
        print(gpr_energy.kernel_,file=fl)
        print(gpr_energy.log_marginal_likelihood(),file=fl)
    return gpr_energy,gpr_density    

# def get_deepmd_energy(atoms):
#     calc = DP(model="EMFF-2025_V1.0.1.pb")
#     atoms.calc = calc
#     calc.set(tmp_dir='./')
#     calc.set(keep_tmp_files=False)
#     calc.set(keep_alive=False)
#     return atoms.get_potential_energy()
def get_gulp_energy_data(atoms,ncpu=8):
    atoms = opt(atoms=atoms,step=1000,l=1,t=0.000001,n=ncpu, lib='reaxff_nn')              ## compute feature
    write_gulp_in(atoms,runword='gradient nosymmetry conv qite verb',lib='reaxff_nn')   ## compute feature
    if ncpu==1:
       subprocess.call('gulp<inp-gulp>out',shell=True)
    else:
       subprocess.call('mpirun -n {:d} gulp<inp-gulp>out'.format(ncpu),shell=True)         ## compute feature
    e = get_reax_energy(fo='out')
    masses  = np.sum(atoms.get_masses())
    volume  = atoms.get_volume()
    density = masses/volume/0.602214129
    return atoms,e,density
    
def get_gulp_energy(atoms,n=4,libfile='reaxff_nn.lib'):
    ''' get the energy of the atoms using gulp, after structure optimization '''
    runword= 'opti conp qiterative stre atomic_stress'
    write_gulp_in(atoms,runword=runword,
                  maxcyc=1000,pressure=0.0,
                  lib='reaxff_nn.lib')
    print('\n-  running gulp optimize ...')
    if n==1:
       system('gulp<inp-gulp>gulp.out')
    else:
       system('mpirun -n {:d} gulp<inp-gulp>gulp.out'.format(n))
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
    writeLammpsIn(log='lmp.log',timestep=0.1,total=3000,restart=None,
              dump_interval=10,
              species=['C','O','N','H'],
              pair_style = pair_style,  # without lg set lgvdw no
              pair_coeff = pair_coeff,
              fix = ' ',
              fix_modify = ' ',
              minimize   = '1e-5 1e-5 3000 3000',
              thermo_style ='thermo_style  custom step temp epair etotal press vol cella cellb cellc cellalpha cellbeta cellgamma pxx pyy pzz pxy pxz pyz',
              data='data.lammps',units=units,atom_style=atom_style,
              restartfile='restart')
    print('\n-  running lammps minimize ...')
    if n==1:
       system('lammps<in.lammps>out')
    else:
       system('mpirun -n {:d} lammps -i in.lammps>out'.format(n))
    atoms = lammpstraj_to_ase('lammps.trj',inp='in.lammps',units=units)
    return atoms

def trajplot(traj='siesta.traj',n=8,i=0,j=1,n_struct=100):
    images_        = Trajectory(traj)
    step,e1,e2,e_dft= [],[],[],[]
    e3,e4          = [],[]
    r              = []
    # 随机抽样100个结构
    n_images = len(images_)
    masses   = np.sum(images_[0].get_masses())
    if n_images > n_struct:
        sampled_indices = random.sample(range(n_images), n_struct)
        images = [images_[idx] for idx in sampled_indices]
    else:
        images = images_

    root_dir   = getcwd()

    data = np.loadtxt('data/feature_mlp.csv',delimiter=',',skiprows=1)      ## get crystal feature data
    data_= np.loadtxt('data/feature.csv',delimiter=',',skiprows=1)          ## get crystal feature data
    X_raw  = data[:,1:]
    y      = data_[:,8]
    y_eng  = data_[:,1]
    scaler = preprocessing.StandardScaler().fit(X_raw)
    X      = scaler.transform(X_raw)
    gpr_energy,gpr_density = load_gaussian_process(X,y,y_eng)

    fe = open("energy.txt","w")
    for i_,atoms in enumerate(images):
        step.append(i_)
        e_dft.append(atoms.get_potential_energy())

        volume = atoms.get_volume()
        density = masses/volume/0.602214129
        
        atoms_rnn = get_gulp_energy(atoms,n=8,libfile='reaxff_nn.lib')
        #### calculate density 
        volume = atoms_rnn.get_volume()
        density_rnn = masses/volume/0.602214129

        e1.append(atoms_rnn.get_potential_energy())
        # e2.append(get_lammps_forces(atoms).get_potential_energy())
        atoms_mtp = get_lammps_energy_mtp(atoms_rnn,n=8)

        volume = atoms_mtp.get_volume()
        density_mtp = masses/volume/0.602214129
        
        e3.append(atoms_mtp.get_potential_energy())

        # e4.append(get_deepmd_energy(atoms))
        ################ get feature and Gaussian corrections #############
        data_dir = '{:s}/data'.format(root_dir)
        chdir(data_dir)
        # print('change to data dir:',data_dir)
        atoms_,e,density_ = get_gulp_energy_data(atoms_rnn,ncpu=n)
        feature = np.array([e[0],e[1],e[5],e[8],e[10],e[11],e[12],density_])
        X_ = scaler.transform(np.expand_dims(feature,axis=0))
        mean_prediction, std_prediction = gpr_density.predict(X_, return_std=True)
        mean_eng_pred, std_eng_pred = gpr_energy.predict(X_, return_std=True)
        chdir(root_dir)
        print(e_dft[-1],e1[-1],e3[-1],density,density_rnn,density_mtp,mean_prediction[0],mean_eng_pred[0],
              std_prediction[0],std_eng_pred[0],file=fe)
        
    fe.close()
    e1  = np.array(e1) - min(e1)
    # e2  = np.array(e2) - min(e2)
    e3  = np.array(e3) - min(e3)
    # e4  = np.array(e4) - min(e4)
    e_dft  = np.array(e_dft)  - min(e_dft)
    sort_idx = np.argsort(e_dft)
    # e   = e[sort_idx]
    # e1  = e1[sort_idx]
    # e2  = e2[sort_idx]
    # e3  = e3[sort_idx]

    if i==0 and j==0:
       r = [i_ for i_ in range(len(e_dft))]
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
    
    plt.plot(r,e_dft,alpha=0.8,
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

