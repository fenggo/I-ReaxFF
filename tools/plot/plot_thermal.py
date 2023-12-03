#!/usr/bin/env python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_thermal():
    plt.figure()
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Step$')
    
    vs = ['2.5','2.8','3.0']
    cs = ['r','b','g']

    for i,v in enumerate(vs):
        fil   = 'thermo-{:s}.log'.format(v)
        data  = np.loadtxt(fil)
        steps = data[:,0]
        e     = data[:,2]
    
        # plt.scatter(ph[i],vh[i],marker = 'o', color = cmap.to_rgba(t), s=50, alpha=0.4)
        plt.plot(steps,e,label=r'$energy .vs. step$', color=cs[i], linewidth=1.5, 
                 linestyle='-')
  
    plt.legend()
    plt.savefig('energy.pdf') 
    plt.close()

 
 
if __name__=='__main__':
   plot_thermal()

