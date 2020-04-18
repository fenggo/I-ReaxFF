#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from ase import Atoms
from ase.io import read,write
 

# 
# data = np.arange(24).reshape((8, 3))
atoms      = read('siesta.traj',index=-1)
positions  = atoms.get_positions()
sym        = atoms.get_chemical_symbols()

 
element = ['C','H','N','O']
color   = {'C':'k','H':'w','O':'r','N':'b'}
size    = {'C':280,'H':160,'O':276,'N':272}

# plot scatter points
fig = plt.figure()
ax = Axes3D(fig)
for elem in element:
    x_,y_,z_ = [],[],[]
    for i,atom in enumerate(atoms):
        if sym[i]==elem:
           x_.append(atom.x)
           y_.append(atom.y)
           z_.append(atom.z)
    ax.scatter(x_, y_, z_, c=color[elem],s=size[elem],label=elem)
 
 
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()
 

