#!/usr/bin/env python
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff_np import IRFF_NP


class Stack():
    def __init__(self,entry=[]):
        self.entry = entry
        
    def push(self,x):
        self.entry.append(x) 

    def pop(self):
        return self.entry.pop()
    
    def close(self):
        self.entry = None


def plot_indiv(findi='Individuals'):
    ids       = []
    id_       = []
    gene      = {}
    density   = []
    dens      = []
    id_g      = {}
    dens_g    = {}
    ent_g     = {}
    op_g      = {}

    with open(findi) as f:
         for line in f.readlines():
             st = Stack([])
             for x in line:
                if x!=']':
                    st.push(x)
                else:
                    x_ = ' '
                    while x_ !='[':
                        x_ = st.pop()
             line = ''.join(st.entry)
             l = line.split()
             
             if len(l)>=10:
                if l[0] != 'Gen':
                   g = l[0]
                   i = int(l[1])
                   e = float(l[3])
                   d = float(l[5])
                   s = l[6]
                   if s=='N/A' or s=='100000.000':
                      continue
                   if g in gene:  
                      gene[g].append(d)
                      id_g[g].append(i)
                      dens_g[g].append(d)
                      ent_g[g].append(e)
                      op_g[g].append(l[2])
                   else:
                      gene[g]    = [d]
                      id_g[g]    = [i]
                      dens_g[g]  = [d]
                      ent_g[g]   = [e]
                      op_g[g]    = [l[2]]
         st.close()
    
    ng      = str(len(gene))
    ops     = {}
    density = {}
    enthalpy= {}
    for i,op in enumerate(op_g[ng]):
        if op not in density:
           density[op]  = [dens_g[ng][i]]
           enthalpy[op] = [ent_g[ng][i]]
        else:
           density[op].append(dens_g[ng][i])
           enthalpy[op].append(ent_g[ng][i])

    plt.figure()
    plt.ylabel(r'$Enthalpy$ ($eV$)',fontdict={'size':10})
    plt.xlabel(r'$Density$ ($g/cm^3$)',fontdict={'size':10})

    markers = {'Heredity':'o','keptBest':'s','softmutate':'^',
               'Rotate':'v','Permutate':'p','Random':'8'}
    colors  = {'Heredity':'#1d9bf7','keptBest':'#c65861','softmutate':'#ffa725',
               'Rotate':'#be588d','Permutate':'#35a153','Random':'#f26a11'}
    for op in density:
        plt.scatter(density[op],np.array(enthalpy[op])+90,alpha=0.9,
                marker=markers[op],color=colors[op],
                edgecolor=colors[op],s=25,
                label=op,
                )
    
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    plt.savefig('individuals.pdf',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''
   plot_indiv(findi='Individuals')


