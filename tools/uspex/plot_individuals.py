#!/usr/bin/env python
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff_np import IRFF_NP
from irff.tools.load_individuals import load_density_energy

I,D,E,O = load_density_energy('Individuals')
lab,e,d = [],[],[]

with open('exp.dat','r') as f:
     for line in f.readlines():
         l = line.split()
         if len(l)>0:
            lab.append(l[0])
            e.append(float(l[1]))
            d.append(float(l[2]))

ops     = {}
density = {}
enthalpy= {}
for i,op in enumerate(O):
    if op not in density:
        density[op]  = [D[i]]
        enthalpy[op] = [E[i]]
    else:
        density[op].append(D[i])
        enthalpy[op].append(E[i])

plt.figure()
plt.ylabel(r'$Enthalpy$ ($eV$)',fontdict={'size':10})
plt.xlabel(r'$Density$ ($g/cm^3$)',fontdict={'size':10})

markers = {'Heredity':'o','keptBest':'s','softmutate':'^',
            'Rotate':'v','Permutate':'8','Random':'p'}
colors  = {'Heredity':'#1d9bf7','keptBest':'#9933fa','softmutate':'#00ffff',
            'Rotate':'#be588d','Permutate':'#35a153','Random':'#00c957'}
r_p = {'520':0.02,'560':0.02,'556':0.02}
hide= []
for op in density:
    # if op in hide:
    #    continue
    plt.scatter(density[op],np.array(enthalpy[op]),alpha=0.9,
            marker=markers[op],color='none',
            edgecolor=colors[op],s=50,
            label=op)
    
for l_,e_,d_ in zip(lab,e,d):
    c = 'k'
    if l_=='epsilon':
       lb = r'$\varepsilon$-CL-20'
       c  = 'r'
    elif l_=='gamma':
       lb = r'$\gamma$-CL-20'
       c = '#4169e1' #0787c3
    elif l_=='beta':
       lb = r'$\beta$-CL-20'
       c = '#2ec0c2'
    plt.scatter(d_,e_,alpha=0.9,
            marker='*',color='none',
            edgecolor=c,s=120,
            label=lb)
    plt.text(d_+0.03,e_,lb,ha='center',color=c,
             fontweight='bold',fontsize=6)

for i,t in enumerate(I):
    d = D[i]
    e = E[i]
    if t in r_p:
       plt.text(d+r_p[t],e,t,ha='center',fontsize=6)
    else:
       plt.text(d,e+0.15,t,ha='center',fontsize=6)

plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
plt.savefig('individuals.svg',transparent=True) 
plt.close() 

