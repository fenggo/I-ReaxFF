#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance
import matplotlib.pyplot as plt
import numpy as np


def get_array(x):
	return np.array(x) - np.min(x)

e_nm,e_h1,e_h2 = [],[],[]
r_nm,r_h1,r_h2 = [],[],[]

images = Trajectory('md-nm.traj')
for i,atoms in enumerate(images):
    e_nm.append(atoms.get_potential_energy())
    v = atoms.positions[0]-atoms.positions[1]
    r_nm.append(np.sqrt(np.sum(np.square(v))))
e_nm = get_array(e_nm)

images = Trajectory('md-hmx1.traj')
for i,atoms in enumerate(images):
    e_h1.append(atoms.get_potential_energy())
    v = atoms.positions[0]-atoms.positions[1]
    r_h1.append(np.sqrt(np.sum(np.square(v))))
e_h1 = get_array(e_h1)

images = Trajectory('md-hmx2.traj')
for i,atoms in enumerate(images):
    e_h2.append(atoms.get_potential_energy())
    v = atoms.positions[0]-atoms.positions[1]
    r_h2.append(np.sqrt(np.sum(np.square(v))))
e_h2 = get_array(e_h2)

plt.figure()   
plt.ylabel(r'$Total$ $Energy$ ($eV$)')
plt.xlabel(r'$Radius$ $(\AA)$')

plt.plot(r_nm,e_nm,alpha=0.9,
         linestyle='-',# marker='o',markerfacecolor='k',markersize=5,
         color='k',label='Nitromethane')

plt.plot(r_h1,e_h1,alpha=0.9,
         linestyle='-',# marker='o',markerfacecolor='k',markersize=5,
         color='b',label=r'$NO_2N(CH_3)_2$')

plt.plot(r_h2,e_h2,alpha=0.9,
         linestyle='-',# marker='o',markerfacecolor='k',markersize=5,
         color='r',label=r'$NO_2NCH_3CH_2NH_2$')

plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
plt.savefig('BondEnergy.pdf',transparent=True) 
plt.close() 
