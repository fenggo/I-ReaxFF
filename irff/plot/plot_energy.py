#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import Trajectory,TrajectoryReader
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff import IRFF



def plot_energy(traj='md.traj'):
    images = Trajectory(traj)
    e,ei,ei_,diff = [],[],[],[]

    # ir_mpnn = IRFF(atoms=images[0],
    #              libfile='ffield.json',
    #              nn=True)
    # ir_mpnn.get_potential_energy(images[0])

    #ir_reax = IRFF(atoms=images[0],
    #             libfile='ffield',
    #             nn=False)
    #ir_reax.get_potential_energy(images[0])
    #atoms = read(traj,index=-1)
    #print(atoms.get_potential_energy())
    #exit()
    
    images = Trajectory('md.traj')
    v = []
    for i,atoms in enumerate(images):
        ei.append(atoms.get_potential_energy())
        # ei.append(ir_mpnn.get_potential_energy(atoms))
        #ei_.append(ir_reax.get_potential_energy(atoms))
        #diff.append(abs(e[-1]-ei[-1]))
        #print(' * energy: ',e[-1],ei[-1],ei_[-1],diff[-1])
        v.append(atoms.get_volume())
        # stress = atoms.get_stress()
        # print(stress)
        del atoms

    #print(' * mean difference: ',np.mean(diff))
    #e_min = min(e)
    #e_max = max(e)
    #e = np.array(e) - e_min
    e_min = min(ei)
    ei = np.array(ei) - e_min

    #e_min = min(ei_)
    #ei_ = np.array(ei_) - e_min

    plt.figure()   
    plt.ylabel(r'$Total$ $Energy$ ($eV$)')
    plt.xlabel(r'$MD$ $Step$')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    plt.plot(ei,alpha=0.9,
             linestyle='-',# marker='o',markerfacecolor='k',markersize=5,
             color='k',label='ReaxFF-MPNN')

    # plt.plot(v,ei,alpha=0.9,
    #          linestyle='-',marker='^',markerfacecolor='none',
    #          markeredgewidth=1,markeredgecolor='blue',markersize=5,
    #          color='blue',label='IRFF')
    # plt.text( 0.0, e_max, '%.3f' %e_min, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
    plt.savefig('Energy.pdf',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [e])
   argh.dispatch(parser)
