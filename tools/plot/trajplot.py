#!/usr/bin/env python
import argparse
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff_np import IRFF_NP
import sys
import argparse


def trajplot(traj='siesta.traj',nn=True,i=0,j=1):
    ffield = 'ffield.json' if nn else 'ffield'
    images = Trajectory(traj)
    step,e,ei      = [],[],[]
    r              = []

    ir = IRFF_NP(atoms=images[0],
              libfile=ffield,
              nn=nn)

    ir.calculate(images[0])

    for i_,atoms in enumerate(images):
        step.append(i_)
        e.append(atoms.get_potential_energy())
        ir.calculate(atoms)
        ei.append(ir.E)
        r.append(ir.r[i][j])

    e_max = min(e)
    e = np.array(e) - e_max
    e_max = min(ei)
    ei   = np.array(ei) - e_max

    plt.figure()   
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Time$ $Step$ ($fs$)')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    # ax = plt.gca()
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.spines['left'].set_position(('data',0))
    # ax.spines['bottom'].set_position(('data', 0))

    plt.plot(r,e,alpha=0.8,
             linestyle='-',#marker='s',markerfacecolor='none',
             #markeredgewidth=1,markeredgecolor='r',markersize=3,
             color='red',label=r'$DFT$ ($SIESTA$)')

    plt.plot(r,ei,alpha=0.8,
             linestyle='-',marker='o',markerfacecolor='none',
             markeredgewidth=1,markeredgecolor='b',markersize=10,
             color='k',label=r'$ReaxFF-nn$')

    ediff = np.abs(e - ei)
    plt.fill_between(r,ei - ediff, ei + ediff, color='palegreen',
                     alpha=0.2)

    # pdiff = np.abs(pdft - preax)
    # plt.fill_between(v_, pdft - pdiff, pdft + pdiff, color='palegreen',
    #                  alpha=0.2)

    plt.text( 0.0, 0.5, '%.3f' %e_max, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # loc = lower left upper right best
    plt.savefig('{:s}'.format(traj.replace('traj','svg')),transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./tplot.py to run it'''

   parser = argparse.ArgumentParser(description='stretch molecules')
   parser.add_argument('--t', default='gulp.traj',type=str, help='trajectory file')
   parser.add_argument('--i', default=0,type=int, help='atom i')
   parser.add_argument('--j', default=1,type=int, help='atom j')
   args = parser.parse_args(sys.argv[1:])
   trajplot(traj=args.t,i=args.i,j=args.j)


