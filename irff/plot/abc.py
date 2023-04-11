#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
from ase.io import read,write
from ase import Atoms
from ase.io.trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt


images = Trajectory('siesta.traj')
a,b,c  = [],[],[]
al,be,ga = [],[],[]

for atoms in images:
    cell = atoms.get_cell()
    r    = np.sqrt(np.sum(cell*cell,axis=1))
    a.append(r[0])
    b.append(r[1])
    c.append(r[2])

    # print(cell[0],cell[1])
    # print(r[0],r[1])

    cosgam = np.dot(cell[0],cell[1])/(r[0]*r[1])
    gam    = np.arccos(cosgam)*180.0/3.14159

    # print('\n',np.dot(cell[0],cell[1]),r[0]*r[1])
    # print('\n',cosgam)

    cosbet = np.dot(cell[0],cell[2])/(r[0]*r[2])
    bet    = np.arccos(cosbet)*180.0/3.14159

    cosalp = np.dot(cell[1],cell[2])/(r[1]*r[2])
    alp    = np.arccos(cosalp)*180.0/3.14159

    # print(alp,bet,gam)
    al.append(alp)
    be.append(bet)
    ga.append(gam)


plt.figure()             # abc
plt.ylabel(r"lattice constant")
plt.xlabel(r"abc (unit:$\AA$)")

plt.plot(a,alpha=0.5,color='blue',linestyle='-',
         label=r"a")
plt.plot(b,alpha=0.5,color='r',linestyle='-',
         label=r"b" )
plt.plot(c,alpha=0.5,color='k',linestyle='-',
         label=r"c" )

plt.legend(loc='best',edgecolor='yellowgreen')
plt.savefig('abc.eps') 
plt.close() 


plt.figure()             # angels
plt.ylabel(r"lattice constant")
plt.xlabel(r"Angels (unit: Degree)")

plt.plot(al,alpha=0.5,color='blue',linestyle='-',
         label=r"$\alpha$")
plt.plot(be,alpha=0.5,color='r',linestyle='-',
         label=r"$\beta$" )
plt.plot(ga,alpha=0.5,color='k',linestyle='-',
         label=r"$\gamma$" )

plt.legend(loc='best',edgecolor='yellowgreen')
plt.savefig('angel.eps') 
plt.close() 

