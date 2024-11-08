#!/usr/bin/env python
from os import getcwd, chdir,listdir
from os.path import isfile,isdir
import matplotlib.pyplot as plt
import numpy as np


data           = np.loadtxt('md_time1.txt')
natom          = data[:,0]  # supercell size
t_gulp_nn      = data[:,1]
t_gulp         = data[:,2]
t_lammps       = data[:,3]
t_lammps_nn    = data[:,4]

plt.figure()

plt.ylabel(r'$Running$ $time$ $of$ $a$ $single$ $MD$ $step$ $Unit:(s)$')
plt.yticks([0.00000000],labels=['     '])

plt.xticks([])

plt.subplot(2,1,1)

# plt.title(r'$Running$ $time$ $of$ $a$ $single$ $MD$ $step$ $Unit:(s)$')


plt.plot(natom,t_gulp,alpha=0.9,
        linestyle='-',marker='^',markerfacecolor='none',
        markeredgewidth=1,markeredgecolor='r',markersize=7,
        color='r',label=r'$ReaxFF(GULP)$' )
plt.plot(natom,t_gulp_nn,alpha=0.9,
        linestyle='-',marker='s',markerfacecolor='none',
        markeredgewidth=1,markeredgecolor='b',markersize=7,
        color='b',label=r'$ReaxFF-nn(GULP)$' )
plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right

plt.subplot(2,1,2)

plt.xlabel(r'$Number$ $of$ $Atoms$')   
plt.plot(natom,t_lammps,alpha=0.9,
        linestyle='-',marker='^',markerfacecolor='none',
        markeredgewidth=1,markeredgecolor='r',markersize=7,
        color='r',label=r'$ReaxFF(LAMMPS)$' )
plt.plot(natom,t_lammps_nn,alpha=0.9,
        linestyle='-',marker='s',markerfacecolor='none',
        markeredgewidth=1,markeredgecolor='b',markersize=7,
        color='b',label=r'$ReaxFF-nn(LAMMPS)$' )
         
plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right



plt.savefig('md_time.pdf',transparent=True) 
plt.close() 

