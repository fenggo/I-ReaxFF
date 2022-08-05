#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff import IRFF


def calculate_pressure(stress):
    nonzero = 0
    stre_   = 0.0
    GPa = 1.60217662*1.0e2
    for _ in range(3):
        if abs(stress[_])>0.0000001:
           nonzero += 1
           stre_   += -stress[_]
    pressure = stre_*GPa/nonzero
    return pressure


def e(traj='pre_dft.traj',batch_size=100,nn=True):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    e,ei,ei_    = [],[],[]

    ir = IRFF(atoms=images[0],
                 libfile=ffield,
                 nn=True)
    ir.get_potential_energy(images[0])

    ir_reax = IRFF(atoms=images[0],
                 libfile='ffield',
                 nn=False)
    ir_reax.get_potential_energy(images[0])

    v_,pdft,pml,preax,diff   = [],[],[],[],[]
    v0  = images[0].get_volume()

    for i,atoms in enumerate(images):
        v=atoms.get_volume()
        e.append(atoms.get_potential_energy())
        stress_ = atoms.get_stress()
        if v/v0 < 0.86:
           pdft.append(calculate_pressure(stress_)) 

           ir.calculate(atoms=atoms,CalStress=True)
           ei.append(ir.E)
           stress  = ir.results['stress']
           pml.append(calculate_pressure(stress)) 

           ir_reax.calculate(atoms=atoms,CalStress=True)
           ei_.append(ir.E)
           stress  = ir_reax.results['stress']
           preax.append(calculate_pressure(stress)) 

           v_.append(v/v0)
           diff.append(abs(pml[-1]-pdft[-1]))
           print(' * V/V0',v_[-1],v,pml[-1],pdft[-1],' diff: ',diff[-1])
     
    print(' * mean error:',np.mean(diff))
    e_min = min(e)
    e_max = max(e)
    e = np.array(e) - e_min
    e_min = min(ei)
    ei = np.array(ei) - e_min

    plt.figure()   
    plt.ylabel(r'$Pressure$ ($GPa$)')
    plt.xlabel(r'$V/V_0$')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    ax   = plt.gca()
    pml  = np.array(pml)
    pdft = np.array(pdft)

    plt.plot(v_,pml,alpha=0.9,
             linestyle='-',marker='o',markerfacecolor='none',markersize=7,
             color='b',label='IRFF(MPNN)')

    plt.plot(v_,preax,alpha=0.9,
             linestyle='-',marker='^',markerfacecolor='none',markersize=7,
             color='g',label='ReaxFF(trained)')

    plt.plot(v_,pdft,alpha=0.9,
             linestyle='-',marker='s',markerfacecolor='none',markersize=7,
             color='r',label='DFT(SIESTA)')
    
    pdiff = np.abs(pdft - pml)
    plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='darkorange',
                     alpha=0.2)

    pdiff = np.abs(pdft - preax)
    plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='palegreen',
                     alpha=0.2)

    plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
    plt.savefig('compare-pv.svg',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [e])
   argh.dispatch(parser)
