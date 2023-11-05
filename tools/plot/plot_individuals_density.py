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
    enthalpy  = []
    gene      = {}
    density   = []
    dens      = []

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
                   # if e>-140.0:
                   if g in gene:  
                      gene[g].append(d)
                   else:
                      gene[g] = [d]
                   id_.append(i)
                   dens.append(d)
                      
                   # enthalpy.append(float(l[3]))
         st.close()
    i_ = 0
    for i in range(1,len(gene)+1):
        #enthalpy.append(min(gene[str(i)]))
        md = max(gene[str(i)])
        for j in gene[str(i)]:
            i_ +=1
            if j>=md:
              density.append(j)
              ids.append(i_)
        # ids.append(i)
        

    plt.figure()
    plt.ylabel(r'$Enthalpy$ ($eV$)')
    plt.xlabel(r'$Generation$')

    plt.plot(ids,density,alpha=0.8,
             linestyle='-',lw=2,
             marker='*',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='r',markersize=15,
             color='r',
             label=r'$max$ $denstiy$ $of$ $generation$',
             )
    plt.scatter(id_,dens,alpha=0.6,
             marker='o',color='none',
             edgecolor='b',s=20,
             label=r'$denstiy$ $of$ $generated$ $crystals$',
             )
    
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    plt.savefig('individuals.svg',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''
   plot_indiv(findi='Individuals')


