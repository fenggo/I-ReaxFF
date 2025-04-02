#!/usr/bin/env python
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory
from irff.tools.load_individuals import load_density_energy
from irff.md.gulp import opt


parser = argparse.ArgumentParser(description='eos by scale crystal box')
parser.add_argument('--g', default=-1,type=int, help='n generation')
parser.add_argument('--o', default=0,type=int, help='optimize structures')
parser.add_argument('--r', default=0,type=int, help='read data from file')
args = parser.parse_args(sys.argv[1:])

n,I_,D_,E_,O_ = load_density_energy('Individuals')
if args.g>0:
   n = args.g
I = I_[n]
D = D_[n]
E = E_[n]
O = O_[n]

if args.o:
   images = Trajectory('Individuals.traj')
   for i,i_ in enumerate(I):
      atoms  = images[int(i_)-1] 
      atoms  = opt(atoms=atoms,step=500,l=1,t=0.0000001,n=16)
      masses = np.sum(atoms.get_masses())
      volume = atoms.get_volume()
      D[i]  = masses/volume/0.602214129
      E[i]  =  atoms.get_potential_energy()
      print(i_,D[i],E[i],O[i])
if args.r:
   with open('indi.dat','r') as f:
      for i,line in enumerate(f.readlines()):
            l = line.split()
            if len(l)>0:
               D[i]  = float(l[1])
               E[i]  = float(l[2]) 

lab,e,d = [],[],[]

if isfile('exp.dat'):
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
hide= ['1538','1497','1475','1521','1532','1535','1530','1527','1525',
       '1475','1521','1482','1477','1536']

for i,op in enumerate(O):
    if I[i] in hide:
       continue
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
right = {'1542':0.02,'1232':0.02,'1496':0.02,'1520':0.02,'174':0.02}
left  = {'1506':-0.03,'1544':-0.03,'1543':-0.03,'1509':-0.03,
         '1519':-0.03,'1534':-0.03}

for op in density:
    #  if op in hide:
    #     continue
    plt.scatter(density[op],np.array(enthalpy[op]),alpha=0.9,
            marker=markers[op],color='none',
            edgecolor=colors[op],s=50,
            label=op)
    
for l_,e_,d_ in zip(lab,e,d):
    c = 'k'
    if l_=='fox7':
       lb = r'$FOX-7(Exp.)$'
       c  = 'r'
    elif l_=='gamma':
       lb = r'$\gamma$-CL-20'
       c = '#4169e1' #0787c3
    elif l_=='beta':
       lb = r'$\beta$-CL-20'
       c = '#2ec0c2'
    else:
       continue
    plt.scatter(d_,e_,alpha=0.9,
            marker='*',color='none',
            edgecolor=c,s=120,
            label=lb)
    plt.text(d_+0.045,e_,lb,ha='center',color=c,
             fontweight='bold',fontsize=6)

hide_text= ['1476','1488','1490','1519','1520','1541','1537','1479']

for i,t in enumerate(I):
    d = D[i]
    e = E[i]
    if t in hide or t in hide_text:
       continue
    if t in right:
       plt.text(d+right[t],e,t,ha='center',fontsize=6)
    elif t in left:
       plt.text(d+left[t],e,t,ha='center',fontsize=6)
    else:
       plt.text(d,e+0.05,t,ha='center',fontsize=6)

plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
plt.savefig('individuals.svg',transparent=True) 
plt.close() 

